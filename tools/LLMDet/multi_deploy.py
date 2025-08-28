import os
import sys
import time
import io
import tempfile
import numpy as np
import torch
import multiprocessing as mp
import matplotlib.pyplot as plt
import uvicorn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager

# # 添加路径
# sys.path.append(os.path.join(os.getcwd(), "LLMDet"))

# 导入 Hugging Face 相关库
from transformers import GroundingDinoProcessor
from modeling_grounding_dino import GroundingDinoForObjectDetection

# 设置多进程启动方法
mp.set_start_method('spawn', force=True)

# 全局变量 - 模型工作进程池
worker_pool = []
worker_index = 0

# 使用 asynccontextmanager 替代 on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker_pool, worker_index
    
    # 检测可用的GPU数量
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print("No GPU devices available, will use CPU")
        gpu_count = 1  # Use CPU
    else:
        print(f"Detected {gpu_count} GPU devices")
    
    # 加载配置
    # 这里可以从环境变量或配置文件加载默认配置
    default_config = {
        "model_id": "fushh7/llmdet_swin_large_hf",  # HF model ID
        "box_threshold": 0.4,
        "text_threshold": 0.3,
    }
    
    # 为每个GPU创建一个模型工作进程
    for gpu_id in range(gpu_count):
        device = f"cuda:{gpu_id}" if gpu_count > 1 else ("cuda:0" if gpu_count == 1 and torch.cuda.is_available() else "cpu")
        default_config["device"] = device
        worker = GroundingDINOWorker(**default_config)
        worker.start()
        worker_pool.append(worker)
    
    print(f"Initialized model worker processes on {len(worker_pool)} devices")
    
    # 应用运行期间的代码
    yield
    
    # 应用关闭时的清理代码
    print("Shutting down all model worker processes")
    for worker in worker_pool:
        worker.stop()
    worker_pool = []

# 使用 lifespan 参数创建 FastAPI 应用
app = FastAPI(
    title="LLMDet API",
    description="API for LLMDet object detection",
    version="1.0.0",
    lifespan=lifespan
)

class GroundingDINOWorker:
    def __init__(self, model_id, box_threshold=0.4, text_threshold=0.3, device="cpu"):
        """初始化模型工作进程"""
        self.model_id = model_id
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        
        self.request_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.process = None
        
    def start(self):
        """启动模型工作进程"""
        self.process = mp.Process(
            target=self._worker_process, 
            args=(
                self.model_id, self.box_threshold, self.text_threshold, 
                self.device, self.request_queue, self.result_queue
            )
        )
        self.process.daemon = True
        self.process.start()
        
    def _worker_process(self, model_id, box_threshold, text_threshold, 
                        device, request_queue, result_queue):
        """工作进程函数，加载模型并处理请求"""
        try:
            # 设置设备
            if device.startswith("cuda:"):
                gpu_id = int(device.split(":")[-1])
                torch.cuda.set_device(gpu_id)
            
            # 初始化 LLMDet 模型
            print(f"Loading LLMDet model to {device}...")
            processor = GroundingDinoProcessor.from_pretrained(model_id)
            model = GroundingDinoForObjectDetection.from_pretrained(model_id).to(device)
            
            print(f"Model loaded to {device}")
            
            # 添加计数器，每处理N个请求就清理一次内存
            request_count = 0
            clean_interval = 10  # 每处理10个请求清理一次

            # 处理请求循环
            while True:
                try:
                    # 获取请求
                    request_id, request_data = request_queue.get()
                    
                    # 处理检测请求
                    image_pil, text_prompt = request_data
                    
                    # 使用 HuggingFace 处理器和模型进行推理
                    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                    
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        box_threshold=box_threshold,
                        text_threshold=text_threshold,
                        target_sizes=[image_pil.size[::-1]]  # [height, width]
                    )[0]  # 取第一个结果（单张图像）
                    
                    # 检查是否检测到物体
                    if len(results["boxes"]) == 0:
                        print(f"No '{text_prompt}' detected.")
                        result = {"message": f"No '{text_prompt}' detected"}
                        result_queue.put((request_id, result))
                        
                        # --- ADDED: Clean up tensors even when no objects are detected ---
                        del inputs, outputs, results
                        # -------------------------------------------------------------
                            
                        continue  # 处理下一个请求
                    
                    # 准备返回的对象信息
                    detected_objects = []
                    
                    # 收集检测到的对象信息
                    boxes = results["boxes"].cpu().numpy()
                    scores = results["scores"].cpu().numpy()
                    labels = results["labels"]
                    
                    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                        detected_objects.append({
                            "id": i + 1,  # 从1开始的序号
                            "label": label,
                            "bbox": box.tolist(),  # [x_min, y_min, x_max, y_max]
                            "confidence": float(score)
                        })
                    
                    # 生成可视化结果
                    result_image = self._generate_visualization(
                        np.array(image_pil), 
                        boxes, 
                        [str(i+1) for i in range(len(boxes))]  # 使用序号作为标签
                    )
                    
                    # 将结果存储为临时文件
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        plt.imsave(tmp.name, result_image)
                        tmp_filename = tmp.name
                            
                    # 返回结果
                    result = {
                        "visualization_path": tmp_filename,
                        "detected_objects": detected_objects
                    }
                        
                    result_queue.put((request_id, result))

                except Exception as e:
                    print(f"{device} worker process error: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    result_queue.put((request_id, {"error": str(e)}))

                finally:
                    # --- ADDED: Explicitly delete tensors and clear cache after every request ---
                    if 'inputs' in locals():
                        del inputs
                    if 'outputs' in locals():
                        del outputs
                    if 'results' in locals():
                        del results

                    import gc; gc.collect()

                    if request_count % clean_interval == 0:
                        torch.cuda.empty_cache()

        except Exception as e:
            print(f"{device} initialization failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _generate_visualization(self, image, boxes, labels):
        """生成可视化结果
        
        参数:
        - image: 原始图像
        - boxes: 边界框
        - labels: 标签（序号）
        """
        # 根据原始图像尺寸来设置图像大小
        h, w = image.shape[:2]
        dpi = 100  # 设置DPI
        figsize = (w/dpi, h/dpi)  # 根据原始图像尺寸和DPI计算figsize
        
        plt.figure(figsize=figsize, dpi=dpi)
        plt.imshow(image)
        
        # 为每个序号创建随机颜色
        label_colors = {}
        for label in labels:
            # 使用随机颜色而不是基于序号的固定颜色
            rgb = np.random.random(3)  # 随机RGB值
            label_colors[label] = np.concatenate([rgb, np.array([1])], axis=0)
        
        # 显示边界框
        for box, label in zip(boxes, labels):
            color = label_colors[label]
            self._show_box(box, plt.gca(), label, color)
        
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0) # 移除边距
        
        # 将图形转换为图像
        fig = plt.gcf()
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        import gc; gc.collect()
        
        return img_array
        
    def _show_box(self, box, ax, label, color):
        """显示框和序号标签"""
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]

        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))
        
        # 显示序号标签，增大字体大小并添加与边界框相同颜色的背景
        text = ax.text(x0, y0-5, label, fontsize=12, color='white', 
                    bbox=dict(facecolor=color, alpha=0.8, pad=2))

    def process_request(self, image_pil, text_prompt):
        """向工作进程提交处理请求并等待结果"""
        request_id = f"{time.time()}_{id(image_pil)}"
        self.request_queue.put((request_id, (image_pil, text_prompt)))
        
        # 等待结果
        while True:
            try:
                result_id, result = self.result_queue.get(timeout=60)  # 60秒超时
                if result_id == request_id:
                    return result
            except Exception as e:
                raise RuntimeError(f"处理请求时超时或发生错误: {str(e)}")
                
    def stop(self):
        """停止工作进程"""
        if self.process and self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()

def get_next_worker():
    """简单的轮询调度获取下一个工作进程"""
    global worker_index, worker_pool
    
    if not worker_pool:
        raise RuntimeError("No available model worker processes")
    
    worker = worker_pool[worker_index]
    worker_index = (worker_index + 1) % len(worker_pool)
    return worker

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    text_prompt: str = Form(...),
    box_threshold: Optional[float] = Form(0.4),
    text_threshold: Optional[float] = Form(0.3)
):
    """
    上传图像和文本提示进行对象检测
    
    参数:
    - file: 上传的图像文件
    - text_prompt: 用于指定要检测的对象的文本描述
    - box_threshold: 检测框置信度阈值
    - text_threshold: 文本匹配阈值
    
    返回:
    - 可视化结果图像和检测到的对象详细信息
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        # 读取上传的图像
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents))
        # 转换为RGB确保格式正确
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        
        # 获取工作进程并处理
        worker = get_next_worker()
        result = worker.process_request(image_pil, text_prompt)
        
        # 检查结果类型
        if "message" in result:
            # 如果未检测到对象，返回 JSON 消消息
            return JSONResponse(content=result, status_code=200)
        
        if "error" in result:
            raise Exception(result["error"])
        
        # 返回可视化结果和检测到的对象信息
        visualization_path = result["visualization_path"]
        detected_objects = result.get("detected_objects", [])
        
        return JSONResponse(content={
            "visualization_path": visualization_path,
            "detected_objects": detected_objects
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_count = torch.cuda.device_count()
    active_workers = len([w for w in worker_pool if w.process and w.process.is_alive()])
    
    return {
        "status": "healthy" if active_workers == len(worker_pool) else "degraded",
        "gpu_count": gpu_count,
        "active_workers": active_workers,
        "total_workers": len(worker_pool)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)