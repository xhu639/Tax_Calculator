"""
向量数据库模块
支持FAISS向量索引和增量更新
使用BGE模型进行文本编码
"""

# 在导入其他库之前设置环境变量，避免多进程冲突导致的段错误
import os
os.environ['OMP_NUM_THREADS'] = '1'  # 限制 OpenMP 线程数
os.environ['MKL_NUM_THREADS'] = '1'  # 限制 MKL 线程数
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # 限制 NumExpr 线程数
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # 限制 OpenBLAS 线程数
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS 上的向量库线程限制
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 禁用 CUDA

import json
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import faiss
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)

def _get_default_model_path() -> str:
    """从 config.yaml 读取默认模型路径"""
    default_path = r"D:\Program Files\关税优化\BAAI_bge-large-en"
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                model_path = config.get('model', {}).get('embedding_model', default_path)
                return model_path
    except Exception as e:
        logger.warning(f"无法读取配置文件，使用默认路径: {e}")
    return default_path

class VectorDatabase:
    """
    向量数据库类
    支持FAISS索引、增量更新和高效搜索
    """
    
    def __init__(self, 
                 model_path: str = None,
                 dimension: int = 1024,
                 use_gpu: bool = True,
                 index_type: str = "flat"):
        """
        初始化向量数据库
        
        Args:
            model_path: BGE模型路径（如果为None，则从config.yaml读取）
            dimension: 向量维度
            use_gpu: 是否使用GPU
            index_type: FAISS索引类型 (flat, ivf, hnsw)
        """
        if model_path is None:
            model_path = _get_default_model_path()
        self.model_path = model_path
        self.dimension = dimension
        self.index_type = index_type
        
        # 禁用GPU，使用CPU处理
        self.use_gpu = False
        
        # 强制PyTorch使用CPU（在加载模型之前设置）
        # 设置环境变量，防止transformers自动使用GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        # 强制PyTorch使用CPU张量和float32类型（使用新的API）
        torch.set_default_dtype(torch.float32)
        torch.set_default_device(torch.device('cpu'))
        
        # 数据存储
        self.descriptions = []  # 商品描述列表
        self.metadata = []      # 元数据列表
        self.index = None       # FAISS索引
        self.model = None       # BGE模型
        self.tokenizer = None   # 分词器
        
        # 统计信息
        self.total_vectors = 0
        self.last_update = None
        
        logger.info(f"向量数据库初始化 - 模型: {model_path}, GPU: {self.use_gpu}")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载BGE模型"""
        try:
            logger.info(f"正在加载BGE模型: {self.model_path}")
            
            if os.path.exists(self.model_path) and "bge" in self.model_path.lower():
                # 使用transformers加载BGE模型，显式指定使用CPU
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                # 先加载模型，然后显式移动到CPU
                self.model = AutoModel.from_pretrained(self.model_path)
                # 确保模型在CPU上（即使系统有GPU也强制使用CPU）
                self.model = self.model.to('cpu')
                self.model.eval()
                
                # 使用CPU处理
                logger.info("使用CPU处理")
                
                logger.info("BGE模型加载成功")
            else:
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
                
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """编码文本为向量"""
        if not texts:
            return np.array([])
        
        logger.info(f"开始编码 {len(texts)} 个文本")
        
        # 为BGE模型添加指令前缀
        if "bge-large" in self.model_path.lower():
            # BGE-large-en 使用英文指令前缀
            if "bge-large-en" in self.model_path.lower() or "en" in self.model_path.lower():
                instruction = "Represent this sentence for searching relevant passages: "
            else:
                # 中文模型使用中文前缀
                instruction = "为这个句子生成表示以用于检索相关文章："
            texts = [instruction + text for text in texts]
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
            batch_texts = texts[i:i + batch_size]
            
            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            # 确保所有输入都在CPU上
            inputs = {k: v.to('cpu') for k, v in inputs.items()}
            
            # 生成嵌入
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]标记的嵌入
                embeddings = outputs.last_hidden_state[:, 0]
                # L2归一化
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def _create_index(self, vectors: np.ndarray) -> faiss.Index:
        """创建FAISS索引"""
        logger.info(f"创建FAISS索引 - 类型: {self.index_type}, 维度: {vectors.shape[1]}")
        
        try:
            # 确保向量是连续的内存布局（C风格），避免段错误
            if not vectors.flags['C_CONTIGUOUS']:
                vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            else:
                vectors = vectors.astype(np.float32)
            
            if self.index_type == "flat":
                # 精确搜索
                index = faiss.IndexFlatIP(vectors.shape[1])  # 内积搜索
            elif self.index_type == "ivf":
                # IVF索引（适合大规模数据）
                quantizer = faiss.IndexFlatIP(vectors.shape[1])
                index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], 100)
                index.train(vectors)
            elif self.index_type == "hnsw":
                # HNSW索引（适合高维数据）
                index = faiss.IndexHNSWFlat(vectors.shape[1], 32)
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
            
            # 确保索引使用CPU（faiss-cpu 默认就是CPU，但显式设置更安全）
            # faiss-cpu 版本不需要显式设置，但为了兼容性，我们确保不使用GPU
            
            # 添加向量到索引
            index.add(vectors)
            
            logger.info(f"FAISS索引创建成功，包含 {index.ntotal} 个向量")
            return index
            
        except Exception as e:
            logger.error(f"创建FAISS索引失败: {e}")
            raise
    
    def build_from_csv(self, csv_path: str, save_path: str = None) -> bool:
        """
        从CSV文件构建向量数据库
        
        Args:
            csv_path: CSV文件路径
            save_path: 保存路径（可选）
            
        Returns:
            是否成功构建
        """
        try:
            logger.info(f"从CSV文件构建向量数据库: {csv_path}")
            
            # 读取CSV文件
            df = pd.read_csv(csv_path)
            logger.info(f"读取到 {len(df)} 条记录")
            
            # 提取描述文本
            descriptions = df['description'].fillna('').tolist()
            
            # 构建元数据
            metadata = []
            for _, row in df.iterrows():
                metadata.append({
                    'tariff_code': row.get('tariff_code', ''),
                    'import_duty': row.get('import_duty', ''),
                    'import_excise': row.get('import_excise', ''),
                    'import_vagst': row.get('import_vagst', ''),
                    'export_duty': row.get('export_duty', ''),
                    'unit': row.get('unit', ''),
                    'sitc_code': row.get('sitc_code', ''),
                    'hierarchy_level': row.get('hierarchy_level', 0),
                    'level_name': row.get('level_name', ''),
                    'has_tax_info': row.get('has_tax_info', False),
                    'page_number': row.get('page_number', ''),
                    'extraction_method': row.get('extraction_method', '')
                })
            
            # 编码文本
            vectors = self._encode_texts(descriptions)
            
            # 创建索引（_create_index 内部已经添加了向量）
            self.index = self._create_index(vectors)
            
            # 保存数据
            self.descriptions = descriptions
            self.metadata = metadata
            self.total_vectors = len(descriptions)
            self.last_update = datetime.now()
            
            logger.info(f"向量数据库构建完成 - 总向量数: {self.total_vectors}")
            
            # 保存到文件
            if save_path:
                self.save(save_path)
            
            return True
            
        except Exception as e:
            logger.error(f"构建向量数据库失败: {e}")
            return False
    
    def add_documents(self, descriptions: List[str], metadata: List[Dict] = None) -> bool:
        """
        增量添加文档
        
        Args:
            descriptions: 描述文本列表
            metadata: 元数据列表
            
        Returns:
            是否成功添加
        """
        try:
            if not descriptions:
                return True
            
            logger.info(f"增量添加 {len(descriptions)} 个文档")
            
            # 编码新文档
            new_vectors = self._encode_texts(descriptions)
            
            # 如果索引不存在，创建新索引（_create_index 内部已经添加了向量）
            if self.index is None:
                self.index = self._create_index(new_vectors)
                self.descriptions = []
                self.metadata = []
                self.total_vectors = 0
            else:
                # 索引已存在，添加新向量
                # 确保向量是连续的内存布局（C风格），避免段错误
                if not new_vectors.flags['C_CONTIGUOUS']:
                    new_vectors = np.ascontiguousarray(new_vectors, dtype=np.float32)
                else:
                    new_vectors = new_vectors.astype(np.float32)
                self.index.add(new_vectors)
            
            # 更新数据
            self.descriptions.extend(descriptions)
            if metadata:
                self.metadata.extend(metadata)
            else:
                # 创建默认元数据
                for i in range(len(descriptions)):
                    self.metadata.append({
                        'tariff_code': f'new_{self.total_vectors + i}',
                        'has_tax_info': False
                    })
            
            self.total_vectors += len(descriptions)
            self.last_update = datetime.now()
            
            logger.info(f"增量添加完成 - 总向量数: {self.total_vectors}")
            return True
            
        except Exception as e:
            logger.error(f"增量添加失败: {e}")
            return False
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               filter_tax_info: bool = None,
               filter_hierarchy_level: int = None,
               filter_tariff_code: str = None) -> List[Dict[str, Any]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_tax_info: 过滤有税率信息的条目
            filter_hierarchy_level: 过滤层级级别
            filter_tariff_code: 过滤税则号
            
        Returns:
            搜索结果列表
        """
        if self.index is None:
            logger.warning("向量数据库未初始化")
            return []
        
        try:
            # 编码查询文本
            query_vector = self._encode_texts([query])
            
            # 搜索
            scores, indices = self.index.search(query_vector, top_k)
            
            # 构建结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= len(self.descriptions):
                    continue
                
                # 应用过滤器
                if self._apply_filters(idx, filter_tax_info, filter_hierarchy_level, filter_tariff_code):
                    result = {
                        'description': self.descriptions[idx],
                        'score': float(score),
                        'index': int(idx),
                        **self.metadata[idx]
                    }
                    results.append(result)
            
            logger.info(f"搜索完成 - 查询: '{query}', 结果数: {len(results)}")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    def _apply_filters(self, idx: int, filter_tax_info: bool, filter_hierarchy_level: int, filter_tariff_code: str) -> bool:
        """应用过滤器"""
        if idx >= len(self.metadata):
            return False
        
        metadata = self.metadata[idx]
        
        # 过滤税率信息
        if filter_tax_info is not None:
            if filter_tax_info and not metadata.get('has_tax_info', False):
                return False
            elif not filter_tax_info and metadata.get('has_tax_info', False):
                return False
        
        # 过滤层级级别
        if filter_hierarchy_level is not None:
            if metadata.get('hierarchy_level', 0) != filter_hierarchy_level:
                return False
        
        # 过滤税则号
        if filter_tariff_code:
            if filter_tariff_code not in metadata.get('tariff_code', ''):
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        return {
            'total_vectors': self.total_vectors,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'use_gpu': self.use_gpu,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'model_path': self.model_path
        }
    
    def save(self, save_path: str) -> bool:
        """保存向量数据库"""
        try:
            logger.info(f"保存向量数据库到: {save_path}")
            
            # 规范化路径（确保跨平台兼容）
            save_path = os.path.normpath(save_path)
            
            # 创建保存目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存索引 - 使用 os.path.join 构建路径，确保格式一致
            base_path = os.path.splitext(save_path)[0]
            index_path = os.path.join(os.path.dirname(base_path), 
                                      os.path.basename(base_path) + '_index.bin')
            index_path = os.path.normpath(index_path)
            faiss.write_index(self.index, index_path)
            
            # 保存数据
            data = {
                'descriptions': self.descriptions,
                'metadata': self.metadata,
                'total_vectors': self.total_vectors,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'use_gpu': self.use_gpu,
                'last_update': self.last_update,
                'model_path': self.model_path,
                'index_path': index_path
            }
            
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info("向量数据库保存成功")
            return True
            
        except Exception as e:
            logger.error(f"保存向量数据库失败: {e}")
            return False
    
    def load(self, load_path: str) -> bool:
        """加载向量数据库"""
        try:
            logger.info(f"加载向量数据库: {load_path}")
            
            # 规范化路径（确保跨平台兼容）
            load_path = os.path.normpath(load_path)
            
            if not os.path.exists(load_path):
                logger.error(f"文件不存在: {load_path}")
                return False
            
            # 加载数据
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            
            # 恢复数据
            self.descriptions = data['descriptions']
            self.metadata = data['metadata']
            self.total_vectors = data['total_vectors']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
            self.last_update = data['last_update']
            self.model_path = data['model_path']
            
            # 禁用GPU，使用CPU处理
            self.use_gpu = False

            base_path = os.path.splitext(load_path)[0]
            index_filename = os.path.basename(base_path) + '_index.bin'
            index_path = os.path.join(os.path.dirname(load_path), index_filename)
            index_path = os.path.normpath(index_path)
 
            if not os.path.exists(index_path) and 'index_path' in data:
                saved_index_path = data['index_path']
                # 规范化保存的路径
                saved_index_path = os.path.normpath(saved_index_path)
                
                # 如果是绝对路径，直接尝试
                if os.path.isabs(saved_index_path):
                    if os.path.exists(saved_index_path):
                        index_path = saved_index_path
                else:

                    alt_path1 = os.path.join(os.path.dirname(load_path), 
                                            os.path.basename(saved_index_path))
                    alt_path1 = os.path.normpath(alt_path1)
   
                    alt_path2 = os.path.normpath(saved_index_path)
                    
                    if os.path.exists(alt_path1):
                        index_path = alt_path1
                    elif os.path.exists(alt_path2):
                        index_path = alt_path2
            
            if os.path.exists(index_path):
                # 先加载到CPU
                self.index = faiss.read_index(index_path)
                
                # 使用CPU索引
            else:
                logger.warning(f"索引文件不存在: {index_path}")
                return False
            
            logger.info(f"向量数据库加载成功 - 总向量数: {self.total_vectors}")
            return True
            
        except Exception as e:
            logger.error(f"加载向量数据库失败: {e}")
            return False

def main():
    """主函数，用于测试向量数据库功能"""
    import argparse
    
    parser = argparse.ArgumentParser(description='向量数据库')
    parser.add_argument('action', choices=['build', 'search', 'add'], help='操作类型')
    parser.add_argument('-c', '--csv', help='CSV文件路径')
    parser.add_argument('-s', '--save', help='保存路径')
    parser.add_argument('-l', '--load', help='加载路径')
    parser.add_argument('-q', '--query', help='查询文本')
    parser.add_argument('-k', '--top_k', type=int, default=10, help='返回结果数量')
    parser.add_argument('-g', '--gpu', action='store_true', help='使用GPU')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建向量数据库
    db = VectorDatabase(use_gpu=args.gpu)
    
    if args.action == 'build':
        if not args.csv:
            print("错误: 需要指定CSV文件路径")
            return 1
        
        success = db.build_from_csv(args.csv, args.save)
        if success:
            print("向量数据库构建成功")
        else:
            print("向量数据库构建失败")
            return 1
    
    elif args.action == 'search':
        if not args.load:
            print("错误: 需要指定加载路径")
            return 1
        
        if not db.load(args.load):
            print("加载向量数据库失败")
            return 1
        
        if not args.query:
            print("错误: 需要指定查询文本")
            return 1
        
        results = db.search(args.query, args.top_k)
        print(f"搜索结果 (查询: '{args.query}'):")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['description'][:50]}... (相似度: {result['score']:.4f})")
    
    elif args.action == 'add':
        if not args.load:
            print("错误: 需要指定加载路径")
            return 1
        
        if not db.load(args.load):
            print("加载向量数据库失败")
            return 1
        
        # 这里可以添加增量更新逻辑
        print("增量更新功能待实现")
    
    return 0

if __name__ == "__main__":
    exit(main())
