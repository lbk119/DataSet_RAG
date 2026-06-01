# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

# 提前加载.env配置文件（确保os.getenv能获取到MinIO相关配置）
load_dotenv()


def normalize_minio_endpoint(raw_endpoint: str | None) -> tuple[str | None, bool | None]:
    """
    兼容两种配置写法：
    1. host:port
    2. http(s)://host:port
    MinIO SDK 需要 endpoint 仅包含 host:port，协议通过 secure 单独传入。
    """
    if not raw_endpoint:
        return None, None

    endpoint = raw_endpoint.strip()
    if "://" not in endpoint:
        return endpoint, None

    parsed = urlparse(endpoint)
    normalized = parsed.netloc or parsed.path
    inferred_secure = parsed.scheme.lower() == "https"
    return normalized, inferred_secure


# 定义MinIO对象存储服务配置（与LLMConfig风格一致，字段对应.env配置项）
@dataclass
class MinIOConfig:
    endpoint: str    # MinIO服务地址，仅保留host:port
    access_key: str  # MinIO访问密钥（对应MINIO_ACCESS_KEY）
    secret_key: str  # MinIO秘钥（对应MINIO_SECRET_KEY）
    bucket_name: str # MinIO默认存储桶名（知识库文件专用）
    minio_img_dir: str #Minio存储图片的文件夹
    minio_secure: bool # 是否使用ssl加密 http 还是 https


normalized_endpoint, inferred_secure = normalize_minio_endpoint(os.getenv("MINIO_ENDPOINT"))
secure_env = os.getenv("MINIO_SECURE")


# 实例化MinIO配置对象，自动从.env读取配置并绑定
minio_config = MinIOConfig(
    endpoint=normalized_endpoint,
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    bucket_name=os.getenv("MINIO_BUCKET_NAME"),
    minio_img_dir=os.getenv("MINIO_IMG_DIR"),
    minio_secure=(secure_env == "True") if secure_env is not None else bool(inferred_secure)
)