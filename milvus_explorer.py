#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Milvus Lite数据库查询工具
用于查看本地Milvus数据库文件中的集合和数据，并支持向量相似度搜索
"""

import argparse
import os
from pymilvus import MilvusClient
from tabulate import tabulate
import numpy as np
import json


def main():
    parser = argparse.ArgumentParser(description="Milvus Lite数据库查询工具")
    parser.add_argument(
        "--uri", type=str, default="./milvus.db", help="Milvus Lite数据库文件路径"
    )
    parser.add_argument("--token", type=str, default="root:Milvus", help="认证令牌")
    parser.add_argument("--db", type=str, default="default", help="数据库名称")
    parser.add_argument(
        "--collection", type=str, default=None, help="要查询的集合名称(如果不指定则列出所有集合)"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="限制返回的数据条数(默认10)"
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="数据查询的偏移量(默认0)"
    )
    parser.add_argument(
        "--show_vectors", action="store_true", help="是否显示向量数据"
    )
    parser.add_argument(
        "--vector_preview", type=int, default=5, help="显示向量的前几个元素(默认5)"
    )
    parser.add_argument(
        "--search", action="store_true", help="执行向量相似度搜索"
    )
    parser.add_argument(
        "--search_text", type=str, default=None, help="要搜索的文本内容(需要在环境中有可用的文本嵌入模型)"
    )
    parser.add_argument(
        "--search_vector", type=str, default=None, help="要搜索的向量，格式为JSON数组"
    )
    parser.add_argument(
        "--search_id", type=int, default=None, help="使用指定ID的数据作为搜索向量"
    )
    parser.add_argument(
        "--top_k", type=int, default=5, help="返回前k个最相似的结果(默认5)"
    )
    
    args = parser.parse_args()
    
    # 确保数据库文件存在
    if not os.path.exists(args.uri):
        print(f"错误: 数据库文件 '{args.uri}' 不存在")
        return
        
    # 连接到Milvus Lite
    try:
        client = MilvusClient(uri=args.uri, token=args.token, db_name=args.db)
        print(f"成功连接到Milvus Lite数据库: {args.uri}")
    except Exception as e:
        print(f"连接Milvus Lite数据库失败: {e}")
        return
    
    # 如果指定了搜索操作
    if args.search:
        if not args.collection:
            print("错误: 执行搜索时必须指定集合名称 (--collection)")
            return
            
        try:
            # 检查集合是否存在
            if not client.has_collection(args.collection):
                print(f"错误: 集合 '{args.collection}' 不存在")
                return
                
            # 获取检索向量
            search_vector = None
            
            # 优先使用search_id
            if args.search_id is not None:
                try:
                    result = client.query(
                        collection_name=args.collection,
                        filter=f"id == {args.search_id}",
                        output_fields=["embedding"],
                        limit=1
                    )
                    if result and "embedding" in result[0]:
                        search_vector = result[0]["embedding"]
                        print(f"使用ID {args.search_id} 的向量进行搜索")
                    else:
                        print(f"错误: 未找到ID为 {args.search_id} 的数据或其没有向量")
                        return
                except Exception as e:
                    print(f"获取ID {args.search_id} 的向量时出错: {e}")
                    return
            # 其次使用search_vector
            elif args.search_vector:
                try:
                    search_vector = json.loads(args.search_vector)
                    if not isinstance(search_vector, list):
                        print("错误: 搜索向量必须是JSON数组格式")
                        return
                    print(f"使用提供的向量进行搜索 (维度: {len(search_vector)})")
                except json.JSONDecodeError:
                    print("错误: 搜索向量必须是有效的JSON数组")
                    return
            # 最后尝试使用search_text
            elif args.search_text:
                print(f"错误: 文本搜索需要嵌入模型，此功能尚未实现")
                print(f"提示: 请使用 --search_vector 或 --search_id 参数进行搜索")
                return
            else:
                print("错误: 必须提供搜索向量 (--search_vector) 或 搜索ID (--search_id) 或 搜索文本 (--search_text)")
                return
                
            # 执行向量搜索
            try:
                search_results = client.search(
                    collection_name=args.collection,
                    data=[search_vector],
                    anns_field="embedding",  # 向量字段名
                    param={"metric_type": "L2"},  # 或者 "IP", "COSINE" 等
                    limit=args.top_k,
                    output_fields=["text", "reference", "metadata"]
                )
                
                if not search_results or not search_results[0]:
                    print(f"未找到相似结果")
                    return
                    
                print(f"\n找到 {len(search_results[0])} 个相似结果:")
                
                for i, hit in enumerate(search_results[0]):
                    print(f"\n--- 相似结果 #{i+1} (相似度得分: {hit['score']:.4f}) ---")
                    
                    # 显示文本内容
                    if "text" in hit:
                        text = hit["text"]
                        if len(text) > 100:
                            text = text[:97] + "..."
                        print(f"文本: {text}")
                    
                    # 显示参考信息
                    if "reference" in hit:
                        print(f"参考: {hit['reference']}")
                    
                    # 显示元数据
                    if "metadata" in hit and hit["metadata"]:
                        print("元数据:")
                        for key, value in hit["metadata"].items():
                            print(f"  {key}: {value}")
                            
                    # 显示ID信息
                    if "id" in hit:
                        print(f"ID: {hit['id']}")
                    
            except Exception as e:
                print(f"执行向量搜索失败: {e}")
                import traceback
                traceback.print_exc()
                
            return
                
        except Exception as e:
            print(f"执行搜索操作失败: {e}")
            return
    
    # 如果没有指定集合，则列出所有集合
    if not args.collection:
        try:
            collections = client.list_collections()
            if not collections:
                print("当前数据库中没有集合")
                return
                
            print("\n集合列表:")
            collection_data = []
            for collection in collections:
                try:
                    description = client.describe_collection(collection)
                    # 尝试获取集合中的数据量
                    try:
                        count = "未知"
                        # 一些版本的pymilvus可能支持get_collection_stats方法
                        if hasattr(client, "get_collection_stats"):
                            stats = client.get_collection_stats(collection)
                            if stats and "row_count" in stats:
                                count = stats["row_count"]
                    except Exception:
                        count = "未知"
                    
                    collection_data.append([collection, description.get("description", ""), count])
                except Exception as e:
                    collection_data.append([collection, f"获取信息失败: {e}", "未知"])
            
            print(tabulate(collection_data, headers=["集合名称", "描述", "数据量"], tablefmt="grid"))
        except Exception as e:
            print(f"获取集合列表失败: {e}")
            return
    else:
        # 指定了集合，展示该集合的数据
        try:
            if not client.has_collection(args.collection):
                print(f"集合 '{args.collection}' 不存在")
                return
                
            # 获取集合信息
            description = client.describe_collection(args.collection)
            print(f"\n集合 '{args.collection}' 信息:")
            print(f"描述: {description.get('description', '无')}")
            
            # 查询数据
            try:
                # 获取字段列表
                fields = description.get("fields", [])
                
                # 确定输出字段
                if args.show_vectors:
                    field_names = [field.get("name") for field in fields if field.get("name") != "id"]
                    output_fields = field_names
                else:
                    field_names = [field.get("name") for field in fields if field.get("name") not in ["id", "embedding"]]
                    # 确保要输出的字段存在
                    output_fields = ["text", "reference", "metadata"]
                    output_fields = [field for field in output_fields if field in field_names]
                
                # 始终添加id字段
                if "id" not in output_fields:
                    output_fields.append("id")
                
                if not output_fields:
                    print(f"警告: 集合中没有可显示的字段")
                    return
                
                # 查询数据
                results = client.query(
                    collection_name=args.collection, 
                    filter="",
                    output_fields=output_fields,
                    limit=args.limit,
                    offset=args.offset
                )
                
                if not results:
                    print(f"集合 '{args.collection}' 中没有数据")
                    return
                
                print(f"\n数据 (显示 {args.offset + 1} - {args.offset + len(results)} 条):")
                
                # 显示数据
                for i, item in enumerate(results):
                    print(f"\n--- 数据 #{args.offset + i + 1} ---")
                    
                    # 显示ID信息
                    if "id" in item:
                        print(f"ID: {item['id']}")
                    
                    # 显示文本内容
                    if "text" in item:
                        text = item["text"]
                        if len(text) > 100:
                            text = text[:97] + "..."
                        print(f"文本: {text}")
                    
                    # 显示参考信息
                    if "reference" in item:
                        print(f"参考: {item['reference']}")
                    
                    # 显示元数据
                    if "metadata" in item and item["metadata"]:
                        print("元数据:")
                        for key, value in item["metadata"].items():
                            print(f"  {key}: {value}")
                    
                    # 显示向量数据
                    if args.show_vectors and "embedding" in item:
                        embedding = item["embedding"]
                        if isinstance(embedding, list) or isinstance(embedding, np.ndarray):
                            dim = len(embedding)
                            preview = embedding[:args.vector_preview]
                            print(f"向量数据 (维度: {dim}):")
                            print(f"  前{args.vector_preview}个元素: {preview}")
                            print(f"  范数: {np.linalg.norm(embedding):.4f}")
                        else:
                            print(f"向量数据: {embedding}")
                    
                    # 显示其他字段
                    for field in field_names:
                        if field not in ["text", "reference", "metadata", "embedding", "id"] and field in item:
                            print(f"{field}: {item[field]}")
            except Exception as e:
                print(f"查询数据失败: {e}")
                print(f"错误详情: {type(e).__name__}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"获取集合信息失败: {e}")


if __name__ == "__main__":
    main() 