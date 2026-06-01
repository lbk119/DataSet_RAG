# 导入核心依赖（和其他配置类共用，只需导入一次）
from dataclasses import dataclass
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

# 提前加载.env配置文件（全局执行一次即可，无需重复写）
load_dotenv()


def normalize_milvus_url(raw_url: str | None) -> str | None:
    """
    兼容 Milvus 地址的两种写法：
    1. host:port
    2. http(s)://host:port
    pymilvus 需要显式协议，默认补全为 http。
    """
    if not raw_url:
        return None

    url = raw_url.strip()
    if "://" in url:
        parsed = urlparse(url)
        normalized = parsed.geturl()
        return normalized.rstrip("/")
    return f"http://{url}"

# ===================== 其他配置类（LLM/Embedding）可放在上方，保持原有代码不变 =====================
# ... 你的LLMConfig、EmbeddingConfig代码 ...

# 定义Milvus向量数据库配置类
@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus服务端连接地址
    chunks_collection: str   # 存储切片的集合名称
    entity_name_collection: str  # 预留-实体名称集合
    item_name_collection: str    # 存储文档对应实体类的集合名称

# 实例化Milvus配置对象（和其他配置对象命名风格统一）
milvus_config = MilvusConfig(
    milvus_url=normalize_milvus_url(os.getenv("MILVUS_URL")),
    chunks_collection=os.getenv("CHUNKS_COLLECTION"),
    entity_name_collection=os.getenv("ENTITY_NAME_COLLECTION"),
    item_name_collection=os.getenv("ITEM_NAME_COLLECTION")
)