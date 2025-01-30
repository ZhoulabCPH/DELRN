import glob
import SimpleITK as sitk
import pandas as pd
import os
import threading


def get_bounding_box(img_path, mask_path):
    input_image = sitk.ReadImage(img_path)
    mask_image = sitk.ReadImage(mask_path)
    if input_image.GetSpacing() != mask_image.GetSpacing():
        print('Spacing error: \nget img spacing: {}\n get mask spacing: {}'.format(
            input_image.GetSpacing(), mask_image.GetSpacing()))
    if input_image.GetSize() != mask_image.GetSize():
        print('Size error: \nget img size: {}\n get mask size: {}'.format(
            input_image.GetSize(), mask_image.GetSize()))

    mask_image = mask_image > 0

    # 获取肿瘤区域的标签
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_image)

    # 假设只有一个肿瘤区域，其标签为1
    bounding_box = label_stats.GetBoundingBox(1)  # 返回值为 (x_min, y_min, z_min, x_size, y_size, z_size)

    # Bounding box的坐标和大小
    x_min, y_min, z_min, x_size, y_size, z_size = bounding_box

    # 输出bounding box信息
    # print(f"Bounding Box:\nX: {x_min} to {x_min + x_size}\nY: {y_min} to {y_min + y_size}\nZ: {z_min} to {z_min + z_size}")

    extract_filter = sitk.ExtractImageFilter()

    # 设置提取区域的大小，需要在每个方向上加上起始点的偏移
    size = [0] * input_image.GetDimension()
    size[0] = x_size  # 图像的宽
    size[1] = y_size  # 图像的高
    size[2] = z_size  # 图像的深度

    # 设置提取的起始索引
    start_index = [x_min, y_min, z_min]

    # 配置滤波器
    extract_filter.SetSize(size)
    extract_filter.SetIndex(start_index)

    # 应用滤波器来裁剪图像
    cropped_image = extract_filter.Execute(input_image)
    cropped_mask = extract_filter.Execute(mask_image)

    return cropped_image, cropped_mask, bounding_box


def crop_image_to_max_size(img_path, mask_path, max_width, max_height, max_depth, subtrahend=None, divisor=None):
    input_image = sitk.ReadImage(img_path)
    mask_image = sitk.ReadImage(mask_path)
    # 获取肿瘤当前的bounding box
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_image > 0)
    bounding_box = label_stats.GetBoundingBox(1)  # 假设肿瘤标签为1

    # 计算当前肿瘤的中心点
    x_center = bounding_box[0] + bounding_box[3] / 2
    y_center = bounding_box[1] + bounding_box[4] / 2
    z_center = bounding_box[2] + bounding_box[5] / 2

    # 计算新的bounding box的起始点
    x_min_new = int(max(0, x_center - max_width / 2))
    y_min_new = int(max(0, y_center - max_height / 2))
    z_min_new = int(max(0, z_center - max_depth / 2))

    # 确保新的bounding box不会超出原始图像的边界
    x_min_new = min(x_min_new, input_image.GetSize()[0] - max_width)
    y_min_new = min(y_min_new, input_image.GetSize()[1] - max_height)
    z_min_new = min(z_min_new, input_image.GetSize()[2] - max_depth)

    # 设置新的bounding box的大小
    new_size = [int(max_width), int(max_height), int(max_depth)]

    # 使用ExtractImageFilter来裁剪图像
    extract_filter = sitk.ExtractImageFilter()
    extract_filter.SetSize(new_size)
    extract_filter.SetIndex([x_min_new, y_min_new, z_min_new])

    cropped_image = None
    cropped_mask = None
    # 裁剪图像
    try:
        cropped_image = extract_filter.Execute(input_image)
        cropped_mask = extract_filter.Execute(mask_image)
    except Exception as e:  # 捕获所有异常，你也可以根据需要指定特定类型的异常
        # 打印错误信息
        print(f'在处理第{i}项时发生错误: {e}')
        # 打印特定的错误信息，例如中心坐标和新的最小值
        print(f'error {x_center}, {y_center}, {z_center}')
        print(f'new {x_min_new}, {y_min_new}, {z_min_new}')
    if subtrahend is not None and divisor is not None:
        data = sitk.GetArrayFromImage(cropped_image)
        cropped_image = (data - subtrahend) / divisor
        cropped_image = sitk.GetImageFromArray(cropped_image)


    return cropped_image, cropped_mask


def process_main(img_path, mask_path, os_path, img_save_path, mask_save_path, remark, statistics_root, statistics_file_name):
    if not os.path.exists(img_save_path):
        # If it doesn't exist, create it
        os.makedirs(img_save_path)
    if not os.path.exists(mask_save_path):
        # If it doesn't exist, create it
        os.makedirs(mask_save_path)
    img_path_csv = pd.read_csv(img_path)
    stats = []

    for i,file_path in enumerate(img_path_csv['path'].tolist()):
        print(i)
        print(file_path)

        cropped_image, cropped_mask, bounding_box = get_bounding_box(os_path+file_path, os_path+mask_path+file_path.split('\\')[-1])
        # 保存裁剪后的图像
        sitk.WriteImage(cropped_image, img_save_path + remark + file_path.split('\\')[-1])
        sitk.WriteImage(cropped_mask, mask_save_path + remark + file_path.split('\\')[-1])

        # 计算体积和平均长宽高
        volume = (bounding_box[3] * bounding_box[4] * bounding_box[5])

        avg_dims = [bounding_box[3], bounding_box[4], bounding_box[5]]

        # 将信息添加到列表中
        stats.append({
            'PatientID': file_path.split('\\')[-1].split('.')[0],  # 假设患者ID是文件名的前缀
            'Width': avg_dims[0],
            'Height': avg_dims[1],
            'Depth': avg_dims[2],
            'Volume': volume
        })

        # 将统计信息保存为CSV
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(statistics_root + statistics_file_name, encoding='utf_8_sig', index=False)


def process_main_crop_max(img_path, mask_path, img_save_path, mask_save_path, remark, max_width, max_height, max_depth, subtrahend, divisor):
    if not os.path.exists(img_save_path):
        # If it doesn't exist, create it
        os.makedirs(img_save_path)
    if not os.path.exists(mask_save_path):
        # If it doesn't exist, create it
        os.makedirs(mask_save_path)
    img_paths = glob.glob(os.path.join(img_path, '*'))
    stats = []

    for i,file_path in enumerate(img_paths):
        print(i)
        print(file_path)
        cropped_image, cropped_mask = crop_image_to_max_size(file_path, os.path.join(mask_path, file_path.split('\\')[-1]), max_width, max_height, max_depth, subtrahend=subtrahend, divisor=divisor)
        if cropped_image == None:
            print('continue')
            continue


        # 保存裁剪后的图像
        sitk.WriteImage(cropped_image, os.path.join(img_save_path, remark + file_path.split('\\')[-1]))
        sitk.WriteImage(cropped_mask, os.path.join(mask_save_path, remark + file_path.split('\\')[-1]))


def process_in_thread_max(img_path, mask_path, img_save_path, mask_save_path, remark, max_width, max_height, max_depth, subtrahend, divisor):
    thread = threading.Thread(target=process_main_crop_max, args=(img_path, mask_path, img_save_path, mask_save_path, remark, max_width, max_height, max_depth, subtrahend, divisor))
    thread.start()
    return thread


if __name__ == '__main__':
    # 设置最大线程数
    max_threads = 4
    semaphore = threading.Semaphore(max_threads)

    dir_names = []
    threads = []
    for i in [[50, 50, 50]]:
        save_dir_name = ''
        for dir_name in dir_names:
            img_path = rf''
            mask_path = rf''
            img_save_path = rf''
            mask_save_path = rf''
            thread = process_in_thread_max(img_path, mask_path, img_save_path, mask_save_path,
                                           '', i[0], i[1], i[2], None, None)
            threads.append(thread)

        for thread in threads:
            thread.join()




