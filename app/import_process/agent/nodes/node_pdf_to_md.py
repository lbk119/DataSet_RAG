import os
import sys
import time
import requests
import zipfile
import shutil
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 项目内部库
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config
from app.core.logger import logger # 统一日志工具
from app.utils.format_utils import format_state
from app.utils.path_util import PROJECT_ROOT

# MinerU配置（缓存配置信息）
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_KEY = mineru_config.api_key

def step1_validate_paths(state: ImportGraphState):
    """
    步骤1：校验PDF路径和输出目录
    Args:
    state: 工作流状态对象，需包含pdf_path/local_dir/task_id
    Returns:
    pdf_path_obj: PDF文件的Path对象
    output_dir_obj: 输出目录的Path对象
    Raises:
    FileNotFoundError: 如果PDF文件不存在或路径无效
    NotADirectoryError: 如果输出目录不存在或不是目录
    """
    func_name = sys._getframe().f_code.co_name
    
    # 校验PDF路径
    pdf_path_str = state.get("pdf_path", "")
    local_dir_str = state.get("local_dir", "")
    if not pdf_path_str:
        raise ValueError("PDF路径未提供，请确保state中包含'pdf_path'")
    if not local_dir_str:
        local_dir_str = str(PROJECT_ROOT / "output")  # 默认使用临时目录
        logger.info(f"【{func_name}】输出目录未提供，使用默认目录：{local_dir_str}")

    pdf_path_obj = Path(pdf_path_str)
    output_dir_obj = Path(local_dir_str)

    if not pdf_path_obj.is_file() or not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF文件不存在或路径无效：{pdf_path_str}")
    
    if not output_dir_obj.exists():
        logger.info(f"【{func_name}】输出目录不存在，正在创建：{local_dir_str}")
        output_dir_obj.mkdir(parents=True, exist_ok=True)
    
    if not output_dir_obj.is_dir():
        raise NotADirectoryError(f"输出目录无效，不是一个目录：{local_dir_str}")
    
    return pdf_path_obj, output_dir_obj

def step2_upload_and_poll(pdf_path_obj: Path) -> str:
    """
    步骤2:上传PDF至MinerU并轮询解析结果
    核心流程：配置校验 → 获取上传链接 → 文件上传（含重试） → 任务轮询（直至完成/失败/超时）
    Args:
    pdf_path_obj: PDF文件的Path对象
    Returns:
    zip_url: 解析完成后返回的MD ZIP文件下载URL
    Raises:
    ValueError(配置缺失)、RuntimeError(请求/上传失败)、TimeoutError(任务超时)
    """
    func_name = sys._getframe().f_code.co_name
    
    # 配置校验
    if not MINERU_BASE_URL or not MINERU_API_KEY:
        raise ValueError("MinerU配置缺失，请确保MINERU_BASE_URL和MINERU_API_KEY已正确设置")
    
    # 构造请求头（符合HTTP规范，Bearer鉴权）
    request_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINERU_API_KEY}"
    }
    # 1. 调用批量接口，获取上传Signed URL和任务batch_id
    url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"
    req_data = {
        "files": [{"name": pdf_path_obj.name}],
        "model_version": "vlm" # 官方推荐解析模型
    }
    logger.debug(f"[获取上传链接] 调用接口：{url_get_upload}，请求参数：{req_data}")
    rsp = requests.post(url_get_upload, json=req_data, headers=request_headers)
    # 响应校验：先验HTTP状态，再验业务返回码
    if rsp.status_code != 200:
        raise RuntimeError(f"获取上传链接失败,HTTP状态码：{rsp.status_code}，响应内容：{rsp.text}")
    rsp_json = rsp.json()
    if rsp_json.get("code") != 0:
        raise RuntimeError(f"获取上传链接失败,业务错误码：{rsp_json.get('code')}，消息：{rsp_json.get('message')}")
    
    # 提取核心数据：上传链接和任务唯一标识
    signed_url = rsp_json["data"]["file_urls"][0]
    batch_id = rsp_json["data"]["batch_id"]
    logger.info(f"[获取上传链接] 成功，batch_id：{batch_id}，上传链接已生成")
   # 2. 读取PDF二进制数据，准备上传
    with open(pdf_path_obj, "rb") as f:
        file_data = f.read()
    # 创建Session（复用TCP连接，禁用代理避免签名验证失败）
    session = requests.Session()
    session.trust_env = False  

    # 3. 文件上传，使用PUT方法，设置超时，捕获异常并记录日志
    try:
        put_rsp = session.put(url=signed_url, data=file_data,timeout=60)
        if put_rsp.status_code != 200:
            logger.warning(f"[文件上传] 初始上传失败，HTTP状态码：{put_rsp.status_code}，响应内容：{put_rsp.text}")
            raise RuntimeError(f"文件上传失败，HTTP状态码：{put_rsp.status_code}，响应内容：{put_rsp.text}")
    except Exception as e:
        logger.warning(f"[文件上传] 初始上传异常，错误信息：{e}")
        raise RuntimeError(f"文件上传异常，错误信息：{e}")
    finally:
        session.close()

    # 4. 根据batch_id轮询任务状态，直至完成/失败/超时
    poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
    start_time = time.time()
    timeout_seconds = 600 # 最大超时时间10分钟（适配600页内PDF）
    poll_interval = 3 # 轮询间隔3秒（平衡查询频率和服务端压力）

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            raise TimeoutError(f"任务轮询超时，已超过{timeout_seconds}秒，batch_id：{batch_id}")
        
        try:
            poll_rsp = requests.get(url=poll_url, headers=request_headers)
        except Exception as e:
            logger.warning(f"[任务轮询] 请求异常，错误信息：{e}, 重试间隔：{poll_interval}秒")
            time.sleep(poll_interval)
            continue
        # 处理HTTP响应错误：5xx服务端繁忙则重试，其他错误直接抛出
        if poll_rsp.status_code != 200:
            if 500 <= poll_rsp.status_code < 600:
                logger.warning(f"[任务轮询] 请求失败，HTTP状态码：{poll_rsp.status_code}，响应内容：{poll_rsp.text}")
                time.sleep(poll_interval)
                continue
            else:
                raise RuntimeError(f"任务轮询失败，HTTP状态码：{poll_rsp.status_code}，响应内容：{poll_rsp.text}")
        
        # 解析轮询结果，校验业务状态
        poll_json = poll_rsp.json()
        if poll_json.get("code") != 0:
            raise RuntimeError(f"任务轮询失败,业务错误码：{poll_json.get('code')}，消息：{poll_json.get('message')}")
        
        extract_results = poll_json["data"]["extract_result"]
        if not extract_results:
            logger.debug(f"[任务轮询] 解析结果为空，继续轮询，batch_id：{batch_id}")
            time.sleep(poll_interval)
            continue
        # 解析任务状态，分支处理
        result_item = extract_results[0] 
        status = result_item["state"]
        # 状态1：任务完成，提取ZIP下载链接
        if status == "done":
            elapsed_time = time.time() - start_time
            logger.info(f"[任务轮询] 任务完成，batch_id：{batch_id},总耗时：{int(elapsed_time)}秒")
            zip_url = result_item.get("full_zip_url")
            if not zip_url:
                raise RuntimeError(f"任务完成但未返回ZIP下载链接，batch_id：{batch_id}")
            return zip_url
        # 状态2：任务失败，提取错误信息抛出
        elif status == "failed":
            err_msg = result_item.get("err_msg", "未知错误")
            raise RuntimeError(f"MinerU解析失败，batch_id：{batch_id}，错误信息：{err_msg}")
        # 状态3：处理中，实时打印进度（覆盖当前行）
        else:
            elapsed_time = time.time() - start_time
            logger.debug(f"[任务轮询] 任务状态：{status}，batch_id：{batch_id}，已耗时{int(elapsed_time)}秒，继续轮询...")
            time.sleep(poll_interval)

def step3_download_and_extract(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
    """
    步骤3：下载并解压MD ZIP文件
    核心流程：下载ZIP → 清理旧目录并解压 → 查找MD文件（按优先级） → 重命名统一为PDF同名MD文件
    Args:
    zip_url: MD ZIP文件的下载URL
    output_dir_obj: 输出目录的Path对象
    pdf_stem: 原PDF文件名（不含扩展名），用于生成MD文件夹名称
    Returns:
    md_path: 解压后的MD文件路径
    Raises:
    RuntimeError(下载失败/响应错误/解压失败)
    """
    func_name = sys._getframe().f_code.co_name
    
    #1. 下载解析结果ZIP包，120秒超时适配大文件
    try:
        rsp = requests.get(zip_url, timeout=120)
        if rsp.status_code != 200:
            raise RuntimeError(f"ZIP文件下载失败，HTTP状态码：{rsp.status_code}，响应内容：{rsp.text}")
    except Exception as e:
        logger.error(f"[{func_name}] ZIP文件下载异常，错误信息：{e}")
        raise RuntimeError(f"ZIP文件下载异常，错误信息：{e}")
    
    # 拼接ZIP包保存路径，按PDF名称唯一命名
    zip_path = output_dir_obj / f"{pdf_stem}.zip"
    with open(zip_path, "wb") as f:
        f.write(rsp.content)
    logger.info(f"[{func_name}] ZIP文件下载成功，保存路径：{zip_path}")
    
    #2. 清理旧解压目录并解压ZIP包（避免旧文件干扰，为每个PDF创建专属目录）
    extract_dir = output_dir_obj / pdf_stem
    if extract_dir.exists():
        try:
            shutil.rmtree(extract_dir)
        except Exception as e:
            logger.warning(f"[{func_name}] 清理旧解压目录失败，错误信息：{e}")
    # 重新创建解压目录
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    logger.info(f"[{func_name}] ZIP文件解压成功，解压目录：{extract_dir}")
    
    # 3. 递归查找解压目录下所有MD文件（适配子目录结构）
    md_file_list = list(extract_dir.rglob("*.md"))
    if not md_file_list:
        raise FileNotFoundError(f"解压目录中未找到任何.md格式文件：{extract_dir}")
    logger.info(f"共找到{len(md_file_list)}个MD文件，按优先级匹配目标文件")
    # 4. 按优先级匹配目标MD文件（同名→full.md→第一个，兜底避免流程中断）
    target_md_file = None
    # 优先级1：与PDF纯名称完全同名的MD文件
    for md_file in md_file_list:
        if md_file.stem == pdf_stem:
            target_md_file = md_file
            logger.info(f"匹配到优先级1目标：与PDF同名的MD文件\n{target_md_file.name}")
            break
    # 优先级2：MinerU默认生成的full.md（不区分大小写）
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower() == "full.md":
                target_md_file = md_file
                logger.info(f"匹配到优先级2目标：MinerU默认文件\n{target_md_file.name}")
                break
    # 优先级3：兜底取第一个MD文件
    if not target_md_file:
        target_md_file = md_file_list[0]
        logger.info(f"未匹配到前两级目标，兜底取第一个MD文件\n{target_md_file.name}")
    # 重命名MD文件：统一为PDF纯名称，便于后续流程处理（仅不同名时执行）
    if target_md_file.stem != pdf_stem:
        logger.info(f"开始重命名MD文件，统一为PDF同名：{pdf_stem}.md")
        new_md_path = target_md_file.with_name(f"{pdf_stem}.md")
        try:
            # 将磁盘上的文件进行重命名
            target_md_file.rename(new_md_path)
            # 更新变量引用
            target_md_file = new_md_path
            logger.info(f"MD文件重命名成功：{pdf_stem}.md")
        except OSError as e:
            logger.warning(f"MD文件重命名失败，将使用原文件名继续流程：\n{str(e)}")
    # 转换为字符串绝对路径返回，适配后续仅支持字符串路径的函数
    final_md_path = str(target_md_file.absolute())
    logger.info(f"===== [{pdf_stem}]解析结果处理完成，最终MD文件路径：{final_md_path}\n=====")
    return final_md_path

def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    LangGraph工作流节点：PDF转MD核心处理节点
    核心流程：路径校验 → MinerU上传解析 → 结果下载解压 → 读取MD内容并更新工作流状态
    参数：state-工作流状态对象，需包含pdf_path/local_dir/task_id
    返回：更新后的工作流状态，新增md_path/md_content
    """
    # 动态获取函数名避免硬编码
    func_name = sys._getframe().f_code.co_name
    # 节点启动日志，打印当前工作流状态
    logger.debug(f"【{func_name}】节点启动，\n当前工作流状态：{format_state(state)}")
    # 开始：记录节点运行状态
    add_running_task(state["task_id"], func_name)

    try:
        # 步骤1：校验PDF路径和输出目录
        pdf_path_obj,output_dir_obj = step1_validate_paths(state)
        # 步骤2：上传PDF至MinerU并轮询解析结果
        zip_url = step2_upload_and_poll(pdf_path_obj)
        # 步骤3：解压下载的MD ZIP文件
        md_path = step3_download_and_extract(zip_url, output_dir_obj,pdf_path_obj.stem)
        # 更新工作流状态：记录MD文件路径和内容
        state["md_path"] = md_path
        logger.info(f"【{func_name}】MD文件生成成功，路径：{md_path}")
        # 读取MD文件内容，捕获异常仅警告不终止
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
        except Exception as e:
            logger.warning(f"【{func_name}】MD文件读取失败，错误信息：{e}")
        add_done_task(state["task_id"], func_name)  # 记录当前任务完成状态，便于监控和调度
    except Exception as e:
        logger.error(f"【{func_name}】PDF转MD流程执行失败：{str(e)}", exc_info=True)
        raise e
    finally:
        logger.debug(f"【{func_name}】节点执行完成，\n更新后工作流状态:{format_state(state)}")

    return state


if __name__ == "__main__":
# 单元测试：验证PDF转MD全流程
    logger.info("===== 开始node_pdf_to_md节点单元测试 =====")
    from app.utils.path_util import PROJECT_ROOT
    from app.import_process.agent.state import create_default_state
    logger.info(f"测试获取根地址：{PROJECT_ROOT}")
    test_pdf_name = os.path.join("doc", "hak180产品安全手册.pdf")
    test_pdf_path = os.path.join(PROJECT_ROOT, test_pdf_name)
    # 构造测试状态
    test_state = create_default_state(
        task_id="test_pdf2md_task_001",
        pdf_path=test_pdf_path,
        local_dir=os.path.join(PROJECT_ROOT, "output")
        )
    node_pdf_to_md(test_state)
    logger.info("===== 结束node_pdf_to_md节点单元测试 =====")