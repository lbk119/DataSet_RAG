from langgraph.graph import StateGraph, END ,START
from app.query_process.agent.state import QueryGraphState
# 导入所有节点函数
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp

workflow = StateGraph(QueryGraphState)
# 添加节点
workflow.add_node("node_item_name_confirm", node_item_name_confirm)
workflow.add_node("node_search_embedding", node_search_embedding)
workflow.add_node("node_search_embedding_hyde", node_search_embedding_hyde)
workflow.add_node("node_web_search_mcp", node_web_search_mcp)
workflow.add_node("node_rrf", node_rrf)
workflow.add_node("node_rerank", node_rerank)
workflow.add_node("node_answer_output", node_answer_output)
# 定义边
workflow.add_edge(START, "node_item_name_confirm")

def router_after_item_name_confirm(state: QueryGraphState) -> str:
    if state.get("answer"):
        return "node_answer_output"
    else:
        return "node_search_embedding","node_search_embedding_hyde","node_web_search_mcp"
    
workflow.add_conditional_edges("node_item_name_confirm", 
                               router_after_item_name_confirm,
                        {
                            "node_search_embedding":"node_search_embedding",
                            "node_search_embedding_hyde":"node_search_embedding_hyde",
                            "node_web_search_mcp":"node_web_search_mcp",
                            "node_answer_output":"node_answer_output"
                        })
workflow.add_edge("node_search_embedding", "node_rrf")
workflow.add_edge("node_search_embedding_hyde", "node_rrf")
workflow.add_edge("node_web_search_mcp", "node_rrf")
workflow.add_edge("node_rrf", "node_rerank")
workflow.add_edge("node_rerank", "node_answer_output")
workflow.add_edge("node_answer_output", END)

query_graph = workflow.compile()