"""完整诊断Milvus数据"""
from storage.milvus_client import MilvusClient
import json

client = MilvusClient()

# 获取集合统计
stats = client.get_collection_stats()
print(f"📊 集合统计:")
print(f"   记录总数: {stats['num_entities']}")
print()

# 使用query查询所有数据（不使用向量搜索）
try:
    from pymilvus import Collection

    collection = Collection(client.collection_name)

    # 查询所有记录的source_filename字段
    results = collection.query(
        expr="id > 0",  # 使用非空表达式
        output_fields=["source_filename", "city", "content_type"]
    )

    print(f" 查询到 {len(results)} 条记录\n")

    # 统计文件名分布
    filename_counts = {}
    city_counts = {}
    type_counts = {}

    for record in results:
        filename = record.get('source_filename', '未知')
        city = record.get('city', '未知')
        content_type = record.get('content_type', '未知')

        filename_counts[filename] = filename_counts.get(filename, 0) + 1
        city_counts[city] = city_counts.get(city, 0) + 1
        type_counts[content_type] = type_counts.get(content_type, 0) + 1

    print("📁 各文件记录数（前20个）:")
    for filename, count in sorted(filename_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"   {filename}: {count}条")

    if len(filename_counts) > 20:
        print(f"   ... 还有 {len(filename_counts) - 20} 个文件")

    print(f"\n🏙️  城市分布:")
    for city, count in sorted(city_counts.items(), key=lambda x: -x[1]):
        print(f"   {city}: {count}条")

    print(f"\n📋 内容类型分布:")
    for ctype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {ctype}: {count}条")

    print(f"\n✅ 唯一文件名数量: {len(filename_counts)}")

except Exception as e:
    print(f"❌ 查询失败: {e}")
    import traceback

    traceback.print_exc()
