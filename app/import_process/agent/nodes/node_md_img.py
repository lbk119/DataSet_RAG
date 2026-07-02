import os
import re
import sys
import base64
from pathlib import Path
from urllib.parse import quote
from typing import Dict, List, Tuple
from collections import deque

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[4]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# MinIO相关依赖
from minio import Minio
from minio.deleteobjects import DeleteObject
# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task,add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def normalize_minio_dir(directory: str) -> str:
    """
    统一清理MinIO目录前后缀，避免对象名和URL出现双斜杠。
    """
    return directory.strip().strip("/")

def step1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    步骤1：初始化数据，获取MD核心信息
    - 从状态对象中提取MD文件路径，读取文件内容
    - 构造Path对象，获取图片文件夹路径（默认与MD同名的文件夹）
    :param state: 导入流程全局状态对象，包含md_path等核心参数
    :return: MD内容字符串、MD文件Path对象、图片文件夹Path对象
    """
    md_path_str = state["md_path"]
    if not md_path_str:
        logger.error(f"Task {state['task_id']}: MD文件路径未提供")
        raise ValueError("MD文件路径未提供")
    path_obj = Path(md_path_str)
    if not state["md_content"]:
        with path_obj.open("r",encoding="utf-8") as f:
            md_content = f.read()
    else:
        md_content = state["md_content"]
    # 构造图片文件夹路径
    images_dir = path_obj.parent / "images"
    return md_content, path_obj, images_dir

def is_supported_image(filename: str) -> bool:
    """
    判断文件是否为MinIO支持的图片格式（后缀不区分大小写）
    :param filename: 文件名（含后缀）
    :return: 支持返回True，否则False
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS

def find_image_in_md(md_content: str, image_filename: str, context_len: int = 100) -> List[Tuple[str, str]]:
    """
    查找MD内容中指定图片的所有引用位置，并返回每个位置的上下文文本
    :param md_content: MD文件完整内容
    :param image_filename: 图片文件名（含后缀）
    :param context_len: 上下文截取长度，默认前后各100字符
    :return: 上下文列表，每个元素为(上文, 下文)元组，无匹配则返回空列表
    """
    # 转义图片文件名特殊字符，避免正则语法错误；编译正则提升匹配效率
    # r 全称是 raw string（原始字符串），作用是：告诉 Python 解释器：不要处理字符串里的转义字符（如 \、\n、\t 等），按字面意思解析。
    pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_filename) + r".*?\)")
    results = []
    # 迭代查找所有MD图片标签匹配项
    for m in pattern.finditer(md_content):
        start, end = m.span()
        # 截取匹配位置的上文和下文（防止索引越界）
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]
        # 打印图片上下文，便于调试
        logger.debug(f"图片[{image_filename}]匹配到引用，上文：{pre_text.strip()}")
        logger.debug(f"图片[{image_filename}]匹配到引用，下文：{post_text.strip()}")
        results.append((pre_text, post_text))
    if not results:
        logger.debug(f"MD内容中未找到图片[{image_filename}]的引用")
    return results

# 步骤2：扫描图片文件夹，筛选MD中实际引用的支持格式图片
def step2_scan_images(md_content: str, images_dir: Path) -> List[Tuple[str,str, Tuple[str, str]]]:
    """
    扫描图片文件夹，过滤出「支持格式+MD中实际引用」的图片，组装处理元数据
    :param md_content: MD文件完整内容
    :param images_dir: 图片文件夹路径对象
    :return: 待处理图片列表，每个元素为(图片文件名, 图片完整路径, 图片上下文)元组
    """
    images_info = []
    if not images_dir.exists() or not images_dir.is_dir():
        logger.info(f"图片文件夹不存在，跳过图片扫描：{images_dir.absolute()}")
        return images_info

    for img_file in images_dir.iterdir():
        if img_file.is_file() and is_supported_image(img_file.name):
            # 查找该图片在MD中的所有引用位置，并提取上下文
            context_list = find_image_in_md(md_content, img_file.name)
            image_path_str = str(images_dir / img_file.name)
            if context_list:
                images_info.append((img_file.name, image_path_str, context_list[0]))
                logger.info(f"找到有效图片引用：{img_file.name}，引用次数：{len(context_list)}")
            else:
                logger.info(f"图片[{img_file.name}]未在MD中找到引用，跳过处理")
        else:
            logger.debug(f"文件[{img_file.name}]不是支持的图片格式，跳过")
    logger.info(f"总共找到{len(images_info)}个有效图片引用，准备进行摘要生成和上传处理")
    return images_info

def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片文件编码为Base64字符串（用于多模态大模型输入）
    :param image_path: 图片本地完整路径
    :return: 图片的Base64编码字符串（UTF-8解码）
    """
    with open(image_path, "rb") as img_file:
        base64_str = base64.b64encode(img_file.read()).decode("utf-8")
    logger.debug(f"图片Base64编码完成，文件：{image_path}，编码后长度：{len(base64_str)}")
    return base64_str

def summarize_image_with_llm(image_path: str, root_folder:str,image_content: Tuple[str, str]) -> str:
    """
    调用多模态大模型生成图片内容摘要（适配LangChain工具类，复用项目统一LLM客户端）
    生成的摘要用于Markdown图片标题，严格控制50字以内中文描述
    :param image_path: 图片本地完整路径
    :param root_folder: 文档所属文件夹/主名，为大模型提供上下文
    :param image_content: 图片在MD中的上下文元组，格式(上文文本, 下文文本)
    :return: 图片内容摘要（异常时返回默认值"图片描述"）
    """
    # 将图片编码为Base64，适配多模态大模型输入要求
    base64_image = encode_image_to_base64(image_path)
    try:
        llm_client = get_llm_client(lm_config.lv_model)
        # 加载提示词模板，填充图片上下文和文档主名
        prompt_template = load_prompt(name="image_summary",root_folder=root_folder,image_content=image_content)
        messages = [
                HumanMessage(content=[
                    {"type": "text", "text": prompt_template},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ])
            ]
        # 调用LLM生成摘要
        response = llm_client.invoke(messages)
        summary = response.content.strip().replace("\n", "")
        logger.info(f"图片摘要生成成功，文件：{image_path}，摘要：{summary}")
        return summary
    except LangChainException as e:
            logger.error(f"调用LLM生成图片摘要失败，文件：{image_path}，错误：{str(e)}")
    except Exception as e:
        logger.error(f"处理图片摘要时发生异常，文件：{image_path}，错误：{str(e)}")
    return "图片描述"

def step3_generate_summaries(doc_stem: str, images_info: List[Tuple[str, str,Tuple[str, str]]],requests_per_minute: int = 9) -> Dict[str, str]:
    """
    步骤3：批量为待处理图片生成内容摘要，带API速率限制防止触发大模型限流
    :param doc_stem: 文档文件名（不含后缀），作为大模型prompt上下文
    :param images_info: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :param requests_per_minute: 每分钟最大API请求数，默认9次（按大模型限制调整）
    :return: 图片摘要字典，键：图片文件名，值：图片内容摘要
    """
    summaries = {}
    request_times = deque()  # 记录请求时间戳的队列，用于速率限制
    for img_name, img_path, img_context in images_info:
        # 应用API速率限制，确保每分钟请求数不超过设定值
        apply_api_rate_limit(request_times, requests_per_minute)
        summary = summarize_image_with_llm(img_path, doc_stem,img_context)
        summaries[img_name] = summary
    return summaries

def clean_minio_directory(minio_client: Minio, directory: str):
    """
    幂等性清理MinIO指定目录下的所有旧文件，防止重名文件内容混淆和垃圾文件堆积
    幂等性：多次调用结果一致，无文件时不报错
    :param minio_client: 已初始化的MinIO客户端对象
    :param directory: 需要清理的MinIO目录路径
    """
    bucket_name = minio_config.bucket_name
    objects_to_delete = []
    # 列出目录下所有对象，构造删除列表
    for obj in minio_client.list_objects(bucket_name, prefix=directory, recursive=True):
        objects_to_delete.append(DeleteObject(obj.object_name))
    if objects_to_delete:
        delete_result = minio_client.remove_objects(bucket_name, objects_to_delete)
        for del_err in delete_result:
            logger.error(f"删除MinIO对象失败：{del_err.object_name}, 错误：{del_err.error}")
        logger.info(f"已清理MinIO目录[{directory}]，共删除{len(objects_to_delete)}个对象")
    else:
        logger.info(f"MinIO目录[{directory}]为空，无需清理")

def upload_images_to_minio(minio_client: Minio, upload_dir: str, image_info:List[Tuple[str, str, Tuple[str, str]]]) -> Dict[str, str]:
    """
    批量上传待处理图片至MinIO，返回图片文件名与访问URL的映射关系
    :param minio_client: 初始化完成的MinIO客户端对象
    :param upload_dir: MinIO上传根目录
    :param image_info: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :return: 图片URL字典，键：图片文件名，值：MinIO访问URL
    """
    urls = {}
    for img_file, img_path, _ in image_info:
        # 构造MinIO对象名称
        object_name = f"{upload_dir}/{img_file}" if upload_dir else img_file
        logger.debug(f"构造MinIO对象名称完成：{object_name}")
        # 上传单张图片并获取URL
        if img_url := upload_to_minio(minio_client, img_path, object_name):
            urls[img_file] = img_url
    logger.info(f"图片批量上传完成，成功上传{len(urls)}/{len(image_info)}张图片")
    return urls
def upload_to_minio(minio_client: Minio, local_path: str, object_name: str) ->str | None:
    """
    将单张本地图片上传至MinIO对象存储，并返回公网可访问URL
    :param minio_client: 初始化完成的MinIO客户端对象
    :param local_path: 图片本地完整路径
    :param object_name: MinIO中要存储的对象名称（带目录）
    :return: 图片MinIO访问URL（上传失败返回None）
    """
    try:
        # 上传本地文件至MinIO（fput_object：文件流上传，适合大文件）
        minio_client.fput_object(bucket_name=minio_config.bucket_name, object_name=object_name,file_path=local_path,
        content_type=f"image/{os.path.splitext(local_path)[1][1:]}")
        # 处理路径特殊字符，避免URL解析错误
        object_name = quote(object_name, safe="/")
        # 根据配置选择HTTP/HTTPS协议
        protocol = "https" if minio_config.minio_secure else "http"
        # 构造MinIO基础访问URL
        base_url = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name.strip('/')}"
        # 拼接完整图片访问URL base_url 后面带 / 中间直接两个字符串拼接即可
        img_url = f"{base_url.rstrip('/')}/{object_name.lstrip('/')}"
        logger.info(f"图片上传成功，访问URL：{img_url}")
        return img_url
    except Exception as e:
        logger.error(f"图片上传MinIO失败：{local_path}，错误信息：{str(e)}")
        return None
def merge_summaries_and_urls(summaries: Dict[str, str], urls: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    """
    合并图片摘要字典和URL字典，过滤掉上传失败无URL的图片
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param urls: 图片URL字典，键：图片文件名，值：MinIO访问URL
    :return: 合并后的图片信息字典，键：图片文件名，值：(摘要, URL)元组
    """
    image_info = {}
    # 遍历摘要字典，仅保留有对应URL的图片
    for image_file, summary in summaries.items():
        if url := urls.get(image_file):
            image_info[image_file] = (summary, url)
    logger.info(f"图片摘要与URL合并完成，有效图片信息{len(image_info)}条")
    return image_info

def replace_image_references(md_content: str, image_info: Dict[str, Tuple[str, str]]) -> str:
    """
    替换MD内容中的本地图片引用为MinIO远程引用，并填充图片摘要作为标题
    ![原图描述](local_folder/abc.png)---->![图片的AI摘要](https://minio-server/abc.png)
    :param md_content: 原始MD文件内容
    :param image_info: 图片信息字典，键：图片文件名，值：(摘要, URL)元组
    :return: 替换后的新MD内容
    """
    for image_file, (summary, new_url) in image_info.items():
        # 转义图片文件名特殊字符，构造正则匹配原有MD图片标签
        pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)", re.IGNORECASE)
        # 替换所有匹配的图片标签为新的URL和摘要
        md_content = pattern.sub( f"![{summary}]({new_url})", md_content)
        logger.info(f"替换图片引用完成：{image_file} → {new_url}")
    return md_content
def step4_upload_and_replace(minio_client: Minio, doc_stem: str, images_info:List[Tuple[str, str, Tuple[str, str]]],summaries: Dict[str, str], md_content: str) -> str:
    """
    步骤4：核心流程-图片上传MinIO + 合并摘要&URL + 替换MD图片引用
    完整流程：清理MinIO旧目录 → 批量上传新图片 → 合并摘要和URL → 替换MD内容
    :param minio_client: 初始化完成的MinIO客户端对象
    :param doc_stem: 文档文件名（不含后缀），作为MinIO上传子目录名（按文档隔离）
    :param images_info: 待处理图片列表，元素为(图片文件名, 图片完整路径, 图片上下文)
    :param summaries: 图片摘要字典，键：图片文件名，值：内容摘要
    :param md_content: 原始MD文件内容
    :return: 图片引用替换后的新MD内容
    """
    # 构造MinIO上传目录：配置根目录 + 文档主名（去除空格，避免路径问题）
    minio_img_dir = normalize_minio_dir(minio_config.minio_img_dir)
    doc_dir = doc_stem.replace(" ", "")
    upload_dir = f"{minio_img_dir}/{doc_dir}" if minio_img_dir else doc_dir
    # 步骤1：清理该文档对应的MinIO旧目录，保证幂等性
    clean_minio_directory(minio_client, upload_dir)
    # 步骤2：批量上传图片至MinIO，获取URL映射
    urls = upload_images_to_minio(minio_client, upload_dir, images_info)
    #步骤3：合并图片摘要和URL，过滤上传失败的图片
    info = merge_summaries_and_urls(summaries, urls)
    # 步骤4：替换MD内容中的本地图片引用为MinIO远程引用
    if info:
        new_md_content = replace_image_references(md_content, info)
        return new_md_content
    return md_content

def step5_backup_new_md_file(origin_md_path: str, md_content: str) -> str:
    """
    步骤5：将处理后的MD内容保存为新文件（原文件不变，避免数据丢失）
    新文件命名规则：原文件名 + _new.md（如test.md → test_new.md）
    :param origin_md_path: 原始MD文件完整路径
    :param md_content: 处理后的新MD内容
    :return: 新MD文件的完整路径
    """
    # 构造新文件路径：替换原后缀为 _new.md
    new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"
    # 写入新MD内容（覆盖写入，若文件已存在则更新）
    with open(new_md_file_name, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"处理后MD文件已保存，新文件路径：{new_md_file_name}")
    return new_md_file_name
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    MD文件图片处理核心节点 - 五步法完成图片全流程处理
    核心流程：
    1. 初始化获取MD内容、文件路径、图片文件夹路径
    2. 扫描图片文件夹，筛选MD中实际引用的支持格式图片，(image_file, img_path, context_list(图片上下文))
    3. 调用多模态大模型为图片生成内容摘要
    4. 将图片上传至MinIO，替换MD中本地图片路径为MinIO访问URL，并填充图片摘要
    #![原图描述](local_folder/abc.png)---->![图片的AI摘要](https://minio-server/abc.png)
    5. 备份原MD文件，保存处理后的新MD文件并更新状态
    :param state: 导入流程全局状态对象，包含task_id、md_path、md_content等核心参数
    :return: 更新后的全局状态对象（md_content/md_path为处理后新值）
    """
    func_name = sys._getframe().f_code.co_name
    add_running_task(state["task_id"], func_name)
    # 步骤1：初始化数据，获取MD核心信息
    md_content,path_obj,images_dir = step1_get_content(state)
    state["md_content"] = md_content
    # 无图片文件夹，直接跳过所有图片处理逻辑
    if not images_dir.exists() or not images_dir.is_dir():
        logger.info(f"Task {state['task_id']}: 图片文件夹不存在，跳过图片处理：{images_dir.absolute()}")
        return state
    # 初始化MinIO客户端，失败则终止流程
    minio_client = get_minio_client()
    if not minio_client:
        logger.error(f"Task {state['task_id']}: MinIO客户端初始化失败，无法处理图片上传")
        return state
    
    # 步骤2：扫描并筛选MD中引用的支持格式图片
    # (image_file, img_path, context_list(图片上下文))
    images_info = step2_scan_images(md_content, images_dir)
    if not images_info:
        logger.info(f"Task {state['task_id']}: MD文件中未找到有效图片引用，跳过图片处理")
        return state
    # 步骤3：调用多模态大模型为图片生成内容摘要
    summaries = step3_generate_summaries(path_obj.stem, images_info)
    # 步骤4：上传图片至MinIO，替换MD图片路径并填充摘要
    # ![原图描述](local_folder/abc.png)----->![图片的AI摘要](https://minio-server/abc.png)
    new_md_content = step4_upload_and_replace(minio_client, path_obj.stem, images_info, summaries, md_content)
    state["md_content"] = new_md_content
    # 步骤5：备份原MD文件，保存新MD文件并更新状态
    new_md_path = step5_backup_new_md_file(state["md_path"], new_md_content)
    state["md_path"] = new_md_path
    add_done_task(state["task_id"], func_name)  # 记录当前任务完成状态，便于监控和调度
    logger.info(f"Task {state['task_id']}: 图片处理完成，更新MD路径：{new_md_path}")
    return state

if __name__ == "__main__":
    """本地测试入口：单独运行该文件时，执行MD图片处理全流程测试"""
    from app.utils.path_util import PROJECT_ROOT
    logger.info(f"本地测试 - 项目根目录：{PROJECT_ROOT}")
    # 测试MD文件路径（需手动将测试文件放入对应目录）
    test_md_name = os.path.join(r"output\\hak180产品安全手册", "hak180产品安全手册.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)
    # 校验测试文件是否存在
    if not os.path.exists(test_md_path):
        logger.error(f"本地测试 - 测试文件不存在：{test_md_path}")
        logger.info("请检查文件路径，或手动将测试MD文件放入项目根目录的output目录下")
    else:
    # 构造测试状态对象，模拟流程入参
        test_state = {
    "md_path": test_md_path,
    "task_id": "test_task_123456",
    "md_content": ""
    }
    logger.info("开始本地测试 - MD图片处理全流程")
    # 执行核心处理流程
    result_state = node_md_img(test_state)
    logger.info(f"本地测试完成 - 处理结果状态：{result_state}")
