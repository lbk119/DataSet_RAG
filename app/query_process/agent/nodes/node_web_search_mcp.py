import sys
import json
import asyncio
import os

if __package__ in (None, ""):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from app.utils.task_utils import add_done_task, add_running_task
from app.conf.bailian_mcp_config import mcp_config
from agents.mcp import MCPServerSse, MCPServerStreamableHttp
from app.core.logger import logger

async def call_mcp_streamable(query):
    """
    异步调用百炼MCP搜索服务的核心函数。
    该函数负责初始化MCP客户端，建立SSE连接，调用远程工具，并返回原始结果。
    :param query: 搜索查询词（通常是经过改写后的精准Query）
    :return: MCP返回的原始结果对象 (包含 content, isError 等字段)
    """
    search_mcp = MCPServerStreamableHttp(name="search_mcp", 
                                         params={
                                             "url": mcp_config.mcp_base_url,
                                             "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                                             "timeout": 10
                                         },
                                         max_retry_attempts=3)
    #连接-调用-关闭
    try:
        await search_mcp.connect()
        logger.info(f"成功连接到MCP服务器: {mcp_config.mcp_base_url}")
        response = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": query,"count": 5}
        )
        logger.info(f"成功调用MCP工具，收到响应")
        return response
    finally:
        await search_mcp.cleanup()
        logger.info(f"已关闭与MCP服务器的连接")

def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    function_name = sys._getframe().f_code.co_name
    add_running_task(state["session_id"], function_name, state.get("is_stream", False))
    logger.info(f"- {function_name} - 开始执行")
    if state.get("course_id"):
        logger.info(f"- {function_name} - 课程模式跳过联网搜索，优先使用课程知识库")
        add_done_task(state["session_id"], function_name, state.get("is_stream", False))
        return {
            "web_search_docs": []
        }
    query = state.get("rewritten_query", "")
    try:
        response = asyncio.run(call_mcp_streamable(query))
    except Exception as e:
        logger.error(f"- {function_name} - MCP搜索失败: {e}")
        add_done_task(state["session_id"], function_name, state.get("is_stream", False))
        return {
            "web_search_docs": []
        }
    # {
    #   "isError": false,
    #   "content": [
    #     {
    #       "text": "{\"pages\":[{\"snippet\":\"欢迎来到今天几号!2026年日历表 今天是5月27日 星期三 农历2026年四月十一 今天阳历2026年5月27日 星期三生肖属相马最新资讯2018年巴西节日假期安排 今天阴历2026年四月十一星座双子座黄历中\\\"立券\\\"是什么意思\",\"hostname\":\"今天几号\",\"hostlogo\":\"\",\"title\":\"今天是农历几月几日星期几_今天是什么日子_今天几号\",\"url\":\"https://www.jintianjihao.com/\"},{\"snippet\":\"2026年5月24日 星期日 双子座(阳历) 农历四月初八 丙午〔马〕年 癸巳月 戊戌日「阴历」 2026年05月份日历表 日一二三四五六 1十五2十六 3十七4十八5十九6二十7廿一8廿二9廿三 10廿四11廿五12廿六13廿七14廿八15廿九16三十 17初一18初二19初三20初四21初五22初六23初七 24初八25初九26初十27十一28十二29十三30十四 31十五 <上一月今日黄历下一月> 节日今年的端午节是2026年06月19日(农历/五月初五),还有26天,中国传统节日。 过年现在距离2027年春节还有258天^_^,2027年的春节是阳历2027年2月6日 星期六,农历丁未〔羊〕年 正月初一! 宜趋四相,不将,玉宇,解神,金匮,天财凶煞 宜忌小耗,天贼,地贼 百忌戊不受田 戌不吃犬 芒种距离2026年芒种还有12天2026-06-05 23:48:0412天6时57分27秒 夏至距离2026年夏至还有28天2026-06-21 16:24:1227天23时33分35秒 小暑距离2026年小暑还有44天2026-07-07 09:56:4043天17时6分3秒 大暑距离2026年大暑还有60天2026-07-23 03:12:4859天10时22分11秒 立秋距离2026年立秋还有75天2026-08-07 19:42:2675天2时51分49秒 处暑距离2026年处暑还有91天2026-08-23 10:18:3190天17时27分54秒 春节2月15日至23日2月14日(周六)、2月28日(周六)上班共9天 端午节6月19日至21日与周末连休共3天\",\"hostname\":\"日历查询\",\"hostlogo\":\"\",\"title\":\"日历查询_农历日历,今天是什么日子,今天是农历几月几日,今天几号_在线日历查询\",\"url\":\"https://www.85415.com/\"}],\"request_id\":\"dd38fad2-2f6d-42b4-8545-d0c673741d11\",\"tools\":[{\"result\":\"#万年历\\n\\n##假日\\n全国爱发日\\n\\n##忌\\n探病,结婚,开业\\n\\n##属相\\n马\\n\\n##周几\\n星期四\\n\\n##宜祭祀,结网,捕捉,余事勿取\\n\\n##纪年丙午\\n\\n##农历null\\n\\n##年份和月份四月十二\\n\\n##具体日期2026-05-28\",\"type\":\"calendar\"}],\"status\":0}",
    #       "type": "text"
    #     }
    #   ]
    # }
    web_search_docs = []
    content_items = []
    is_error = False
    if response:
        if isinstance(response, dict):
            is_error = response.get("isError", False)
            content_items = response.get("content", [])
        else:
            is_error = getattr(response, "isError", False)
            content_items = getattr(response, "content", [])

    if response and not is_error:
        try:
            first_item = content_items[0]
            payload_text = first_item["text"] if isinstance(first_item, dict) else getattr(first_item, "text")
            payload = json.loads(payload_text)
            web_search_docs = payload.get("pages", [])
        except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as e:
            logger.error(f"- {function_name} - MCP响应解析失败: {e}")
    logger.info(f"- {function_name} - MCP搜索返回的文档数量: {len(web_search_docs)}")
    add_done_task(state["session_id"], function_name, state.get("is_stream", False))
    return {
        "web_search_docs": web_search_docs
    }

if __name__ == '__main__':
    # 测试代码：单独运行该文件时，验证MCP搜索功能是否正常
    print("\n" + "="*50)
    print("> 启动 node_web_search_mcp 本地测试")
    print("="*50)
    test_state = {
    "session_id": "test_mcp_session",
    "rewritten_query": "HAK 180 在出厂默认状态下，若想在纸张上只把烫金膜转印到顶部50 mm–170 mm 的局部区域，应在操作面板上如何设置",
    "is_stream": False
    }
    try:
        # 调用MCP搜索节点函数，执行测试
        result_state = node_web_search_mcp(test_state)
        print("\n" + "="*50)
        print("> 测试结果摘要:")
        search_results = result_state.get('web_search_docs', [])
        print(f"搜索结果数量: {len(search_results)}")
        if search_results:
            print("首条结果预览:")
            print(json.dumps(search_results[0], indent=2, ensure_ascii=False))
        else:
            print("未获取到搜索结果")
            print("="*50)
    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")


