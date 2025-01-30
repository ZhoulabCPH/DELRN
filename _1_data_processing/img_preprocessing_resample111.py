import SimpleITK as sitk
import numpy as np
import pandas as pd
import os


def get_img_and_mask(img_path, mask_path):
    input_image = sitk.ReadImage(img_path)
    mask_image = sitk.ReadImage(mask_path)
    if input_image.GetSpacing() != mask_image.GetSpacing():
        print('Spacing error: \nget img spacing: {}\n get mask spacing: {}'.format(
            input_image.GetSpacing(), mask_image.GetSpacing()))
    if input_image.GetSize() != mask_image.GetSize():
        print('Size error: \nget img size: {}\n get mask size: {}'.format(
            input_image.GetSize(), mask_image.GetSize()))

    return input_image, mask_image


def process_N4Bias(input_image, mask_image):
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # output_image_without_mask = corrector.Execute(input_image)
    output_image = corrector.Execute(input_image,mask_image)

    output_image = sitk.Cast(output_image, sitk.sitkInt16)
    # sitk.WriteImage(output_image, '../dataset/{}'.format(img_path))
    # sitk.WriteImage(mask_image, '../dataset/{}'.format(img_path))
    return output_image, mask_image


def resample_image_itk(
        ori_img,
        new_spacing=[1.0, 1.0, 1.0],
        new_size=[0, 0, 0],
        resamplemethod=sitk.sitkLinear):
    """
    @Args:
        :param ori_img: 原始需要对齐的itk图像
        :param new_spacing: 111
        :param new_size: 默认000, 注意非零则自动裁剪
        :param resamplemethod:
                sitk.sitkLinear-线性 - image
                sitk.sitkNearestNeighbor-最近邻 -mask
    @Return:
        重采样好的itk图像
    """
    ori_size = ori_img.GetSize()  # 原始图像大小  [x,y,z]
    ori_spacing = ori_img.GetSpacing()  # 原始的体素块尺寸    [x,y,z]
    ori_origin = ori_img.GetOrigin()  # 原始的起点 [x,y,z]
    ori_direction = ori_img.GetDirection()  # 原始的方向 [冠,矢,横]=[z,y,x]
    # print('original img size: {}'.format(ori_size))
    # print('original img spacing: {}'.format(ori_spacing))

    # 计算改变spacing后的size，用物理尺寸/体素的大小
    if new_size == [0, 0, 0]:
        new_size[0] = int(ori_size[0] * ori_spacing[0] / new_spacing[0] + 0.5)
        new_size[1] = int(ori_size[1] * ori_spacing[1] / new_spacing[1] + 0.5)
        new_size[2] = int(ori_size[2] * ori_spacing[2] / new_spacing[2] + 0.5)

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(new_size)  # 目标图像大小
    resampler.SetOutputOrigin(ori_origin)
    resampler.SetOutputDirection(ori_direction)
    resampler.SetOutputSpacing(new_spacing)

    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)  # 近邻插值用于mask的，保存uint16
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    resampler.SetOutputPixelType(ori_img.GetPixelID())
    res_img = resampler.Execute(ori_img)
    # print('resampling img size: {}'.format(res_img.GetSize()))
    # print('resampling img spacing: {}'.format(res_img.GetSpacing()))
    return res_img  # 得到重新采样后的图像


def array2nii(image_array, out_path, NIIimage_resample):
    ## image_array是矩阵，out_path是带文件名的路径，NIIimage_resample是sitk_obj
    # 1.构建nrrd阅读器
    image2 = NIIimage_resample
    # 2.将整合后的数据转为array，并获取dicom文件基本信息
    origin = image2.GetOrigin()  # x, y, z
    spacing = image2.GetSpacing()  # x, y, z
    direction = image2.GetDirection()  # x, y, z
    # 3.将array转为img，并保存为.nii.gz
    image3 = sitk.GetImageFromArray(image_array)
    image3.SetSpacing(spacing)
    image3.SetDirection(direction)
    image3.SetOrigin(origin)
    sitk.WriteImage(image3, out_path)


def get_img_from_arr(img, arr):
    itk_img = sitk.GetImageFromArray(np.transpose(arr, (2, 1, 0)))
    itk_img.SetOrigin(img.GetOrigin())
    itk_img.SetSpacing(img.GetSpacing())
    itk_img.SetDirection(img.GetDirection())
    return itk_img


def get_img_have_mask(img, mask, imgNo):
    img_arr = np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))
    mask_arr = np.transpose(sitk.GetArrayFromImage(mask), (2, 1, 0))

    img_layers_num = img_arr.shape[2]
    mask_layers_num = mask_arr.shape[2]

    try:
        assert img_layers_num == mask_layers_num, '{} img_layers_num doesn\'t equal to mask_layers_num'.format(imgNo)
    except AssertionError as error:
        # 如果断言失败，则会执行这里的代码
        print(error)  # 打印出错信息
        return 'error', 'error'


    sum_over_layers = np.sum(mask_arr, axis=(0, 1))

    # 找出哪些层的和不为0，即含有1的层
    select_layer = np.nonzero(sum_over_layers)[0]

    selected_img = img_arr[:, :, select_layer]
    selected_mask = mask_arr[:, :, select_layer]

    itk_img = get_img_from_arr(img, selected_img)
    itk_mask = get_img_from_arr(mask, selected_mask)


    return itk_img, itk_mask


def process_main(img_pid, img_path, mask_path, img_save_path, mask_save_path, remark, N4Bias=False, window=None):
    if not os.path.exists(img_save_path):
        # If it doesn't exist, create it
        os.makedirs(img_save_path)
    if not os.path.exists(mask_save_path):
        # If it doesn't exist, create it
        os.makedirs(mask_save_path)
    img_path_csv = pd.read_csv(img_pid)
    for i,file_path in enumerate(img_path_csv['path'].tolist()):
        print(i)
        print(file_path)

        img, mask = get_img_and_mask(os.path.join(img_path, file_path.split('\\')[-1]), os.path.join(mask_path, file_path.split('\\')[-1]))

        if N4Bias:
            img, mask = process_N4Bias(img, mask)


        resp_img = resample_image_itk(img,new_spacing=[1.0, 1.0, 1.0],
                                      new_size=[0, 0, 0],
                                      resamplemethod=sitk.sitkLinear)
        resp_mask = resample_image_itk(mask,new_spacing=[1.0, 1.0, 1.0],
                                      new_size=[0, 0, 0],
                                      resamplemethod=sitk.sitkNearestNeighbor)
        if window is not None:
            resp_img = sitk.IntensityWindowing(resp_img, windowMinimum=window[0], windowMaximum=window[1],
                                             outputMinimum=window[0], outputMaximum=window[1])


        sitk.WriteImage(resp_img, os.path.join(img_save_path, remark + file_path.split('\\')[-1]))
        sitk.WriteImage(resp_mask, os.path.join(mask_save_path, remark + file_path.split('\\')[-1]))


if __name__ == '__main__':

    dir_name = ''
    img_pid = r''
    mask_path = r''
    img_path = r''
    img_save_path = rf''
    mask_save_path = rf''
    process_main(img_pid, img_path, mask_path , img_save_path, mask_save_path, '')






