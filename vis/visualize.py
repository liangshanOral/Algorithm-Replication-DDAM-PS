import cv2
import matplotlib.pyplot as plt
import json
import os

def visualize_detection(image_root, results):
    for result in results:
        # 创建子图网格
        fig, axes = plt.subplots(1, 4, figsize=(20, 10))

        query_img_path = image_root + '/' + result['query_img']
        query_roi = result['query_roi']

        # 读取查询图像
        query_img = cv2.imread(query_img_path)
        if query_img is None:
            print(f"Unable to read query image: {query_img_path}")
            continue
        # 获取查询 ROI 的坐标
        x1, y1, x2, y2 = map(int, query_roi)
        # 裁剪查询者图像
        query_img_cropped = query_img[y1:y2, x1:x2]

        # 绘制查询者
        cv2.rectangle(query_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 显示裁剪后的查询者图像和检测结果
        
        axes[0].imshow(cv2.cvtColor(query_img_cropped, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Query Result for {result['query_img']}")
        axes[0].axis('off')

        # 对 gallery 结果按照 score 排序
        sorted_gallery = sorted(result['gallery'], key=lambda x: x['score'], reverse=True)
        top3_gallery = sorted_gallery[:3]

        # 绘制前三个得分排名的图库图像和检测框
        for i,gallery_item in enumerate(top3_gallery):
            gallery_img_path = image_root + '/' + gallery_item['img']
            gallery_roi = gallery_item['roi']

            # 读取图库图像
            gallery_img = cv2.imread(gallery_img_path)
            if gallery_img is None:
                print(f"Unable to read gallery image: {gallery_img_path}")
                continue

            # 绘制检测框
            cv2.rectangle(gallery_img, (int(gallery_roi[0]), int(gallery_roi[1])), (int(gallery_roi[2]), int(gallery_roi[3])), (0, 255, 0), 2)

            # 显示图库图像和检测结果
            axes[i+1].imshow(cv2.cvtColor(gallery_img, cv2.COLOR_BGR2RGB))
            axes[i+1].set_title(f"Detection Result for {gallery_item['img']} (Score: {gallery_item['score']})")
            axes[i+1].axis('off')
            
        plt.tight_layout()
        # 保存绘制的图像
        output_dir='DDAM-PS-main/vis/results_vis'
        output_filename = result['query_img'].split('.')[0] + '_visualized.png'  # 以查询图像的名称作为文件名
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        plt.close()


def load_result(file):
    with open(file,'r') as f:
        data=json.load(f)
    #print(data['results'])
    visualize_detection(image_root, data['results'])
    


# 示例调用

json_file = 'DDAM-PS-main/vis/results.json'
image_root = "F:/reproduction/dataset/CUHK-SYSU/Image/SSM"

load_result(json_file)