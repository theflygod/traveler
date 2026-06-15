"""清理Milvus中的错误集合"""
from pymilvus import connections, utility

print("连接到Milvus...")
connections.connect(uri="http://192.168.10.150:19530")

collection_name = "md_chunks"
if utility.has_collection(collection_name):
    print(f"删除集合: {collection_name}")
    utility.drop_collection(collection_name)
    print("✅ 删除成功")
else:
    print(f"集合 {collection_name} 不存在")

connections.disconnect("default")
print("完成")
