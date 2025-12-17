#!/usr/bin/env python3
"""
本地 SRT 字幕翻译脚本 (Hunyuan-MT-7B)
功能：加载本地 Hunyuan-MT-7B 模型 -> 读取 SRT -> 批量翻译 -> 生成字幕
"""

import os
import sys
import argparse
import logging
import torch
import srt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

# 配置日志
def setup_logging(log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)

class LocalTranslator:
    """本地模型翻译器"""
    
    def __init__(self, args):
        self.input_file = args.input_file
        self.output_file = args.output_file
        self.model_path = args.model_path
        self.device = args.device
        self.batch_size = args.batch_size
        self.bilingual = args.bilingual
        
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """加载 Hunyuan-MT-7B 模型 (支持自动下载)"""
        try:
            logger.info(f"正在准备模型: {self.model_path} (Device: {self.device})...")
            
            # 自动检测设备
            if self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            
            logger.info(f"使用设备: {self.device}")

            # 尝试加载 Tokenizer，如果本地不存在则自动从 Hugging Face 下载
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            except OSError:
                logger.info(f"本地未找到模型，尝试从 Hugging Face 下载: {self.model_path}")
                # 默认使用 tencent/Hunyuan-MT-7B，如果用户传的是路径但不存在，可能用户想用的是这个 ID
                # 这里假设如果路径不存在，就把它当做 model_id 处理
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            # 加载模型
            # Hunyuan-MT-7B 通常是基于 Transformer 结构，可以使用 AutoModelForCausalLM 加载
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                trust_remote_code=True,
                device_map=self.device
            )
            
            self.model.eval()
            logger.info("模型加载成功")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("提示: 如果下载失败，请尝试配置 HF_ENDPOINT 环境变量或手动下载模型。")
            sys.exit(1)

    def load_subtitles(self) -> List[srt.Subtitle]:
        """加载 SRT 文件"""
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                content = f.read()
            return list(srt.parse(content))
        except Exception as e:
            logger.error(f"读取字幕文件失败: {e}")
            sys.exit(1)

    def translate_batch(self, texts: List[str]) -> List[str]:
        """批量翻译"""
        # 构造 Prompt
        # Hunyuan-MT 的 prompt 格式可能需要参考官方文档
        # 这里假设使用标准的翻译指令格式
        
        results = []
        
        # 逐条推理 (Hunyuan-MT 可能针对单句优化，批量生成需要 padding)
        # 为了稳妥，这里演示逐条生成，如果显存够大可以自行改为 batch encoding
        for text in texts:
            prompt = f"你是一名精通日语的翻译官，将下面的日文翻译成简体中文,直接给出答案：\n{text}\n答案："
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                # 移除可能导致警告或错误的 token_type_ids，如果模型不需要它
                if "token_type_ids" in inputs:
                    del inputs["token_type_ids"]
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=512,
                        temperature=0.1, # 翻译建议低温度
                        top_p=0.9
                    )
                
                # 解码并去除 prompt 部分
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # 简单的后处理：提取“答案：”之后的内容
                # 注意：实际输出可能包含 prompt，需要截取
                if "答案：" in generated:
                    translation = generated.split("答案：")[-1].strip()
                else:
                    # 如果模型没有复读 prompt，直接取结果
                    # 很多 Chat 模型 generate 会包含输入，需要看具体实现
                    # 这里做一个简单的长度判断，如果包含 prompt 则截取
                    if generated.startswith(prompt):
                        translation = generated[len(prompt):].strip()
                    else:
                        translation = generated.strip()
                
                results.append(translation)
                
            except Exception as e:
                logger.error(f"翻译失败: {text} -> {e}")
                results.append(text) # 失败保留原文
                
        return results

    def process(self):
        """执行流程"""
        self.load_model()
        
        logger.info(f"读取字幕: {self.input_file}")
        subs = self.load_subtitles()
        
        new_subs = []
        pbar = tqdm(total=len(subs))
        
        # 分批处理
        batch_size = self.batch_size
        for i in range(0, len(subs), batch_size):
            batch_subs = subs[i:i + batch_size]
            batch_texts = [s.content.replace('\n', ' ') for s in batch_subs]
            
            translated_texts = self.translate_batch(batch_texts)
            
            for sub, trans in zip(batch_subs, translated_texts):
                if self.bilingual:
                    new_content = f"{trans}\n{sub.content}"
                else:
                    new_content = trans
                
                new_subs.append(srt.Subtitle(
                    index=sub.index,
                    start=sub.start,
                    end=sub.end,
                    content=new_content
                ))
            
            pbar.update(len(batch_subs))
            
        pbar.close()
        
        # 保存
        output_path = self.output_file
        if not output_path:
            name, ext = os.path.splitext(self.input_file)
            output_path = f"{name}_local_chs{ext}"
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(srt.compose(new_subs))
            
        logger.info(f"翻译完成: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="本地 Hunyuan-MT-7B 字幕翻译工具")
    parser.add_argument("--model_path", "-m", default="tencent/Hunyuan-MT-7B", help="模型名称或本地路径 (默认: tencent/Hunyuan-MT-7B)")
    parser.add_argument("--output_file", "-o", help="输出文件路径")
    parser.add_argument("--device", "-d", default="auto", help="设备 (cpu/cuda/mps/auto)")
    parser.add_argument("--batch_size", "-b", type=int, default=1, help="批处理大小 (建议 1-4，视显存而定)")
    parser.add_argument("--bilingual", action="store_true", help="输出双语字幕")
    
    args = parser.parse_args()
    args.input_file = "HHED-027.ass" 
    if not os.path.exists(args.input_file):
        print(f"文件不存在: {args.input_file}")
        sys.exit(1)
        
    setup_logging()
    translator = LocalTranslator(args)
    translator.process()

if __name__ == "__main__":
    main()
