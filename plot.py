import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, ConnectionPatch
import numpy as np

# 定义颜色
White = 1.
Light = 0.7
Medium = 0.5
Dark = 0.3
Darker = 0.15
Black = 0.

# 定义全局参数
NumDots = 4
NumConvMax = 8
NumFcMax = 20
fc_unit_size = 2
layer_width = 40
flag_omit = True

def add_layer(patches, colors, size=(24, 24), num=5,
              top_left=[0, 0],
              loc_diff=[3, -3],
              ):
    # 添加一个卷积层或全连接层的矩形表示
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    for ind in range(num):
        patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
        if ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)

def add_layer_with_omission(patches, colors, size=(24, 24),
                            num=5, num_max=8,
                            num_dots=4,
                            top_left=[0, 0],
                            loc_diff=[3, -3],
                            ):
    # 添加一个卷积层或全连接层的矩形表示，当元素数量超过num_max时用省略号表示
    top_left = np.array(top_left)
    loc_diff = np.array(loc_diff)
    loc_start = top_left - np.array([0, size[0]])
    this_num = min(num, num_max)
    start_omit = (this_num - num_dots) // 2
    end_omit = this_num - start_omit
    start_omit -= 1
    for ind in range(this_num):
        if (num > num_max) and (start_omit < ind < end_omit):
            omit = True
        else:
            omit = False

        if omit:
            patches.append(
                Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
        else:
            patches.append(Rectangle(loc_start + ind * loc_diff,
                                     size[1], size[0]))

        if omit:
            colors.append(Black)
        elif ind % 2:
            colors.append(Medium)
        else:
            colors.append(Light)

def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
                top_left_list, loc_diff_list, num_show_list, size_list):
    # 添加层与层之间的映射关系
    start_loc = top_left_list[ind_bgn] \
        + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
        + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
                    - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])])

    end_loc = top_left_list[ind_bgn + 1] \
        + (num_show_list[ind_bgn + 1] - 1) * np.array(
            loc_diff_list[ind_bgn + 1]) \
        + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
                    - end_ratio[1] * size_list[ind_bgn + 1][0]])

    patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
    colors.append(Dark)
    patches.append(ConnectionPatch(xyA=start_loc, xyB=end_loc, coordsA="data", coordsB="data",
                                   arrowstyle="-", color=Darker))
    colors.append(Darker)

def label(xy, text, xy_off=[0, 4]):
    plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
             family='sans-serif', size=8)

if __name__ == '__main__':
    patches = []
    colors = []
    fig, ax = plt.subplots()

    ############################
    # 卷积层
    conv_size_list = [(128, 128), (128, 128), (64, 64), (64, 64), (32, 32)]
    conv_num_list = [1, 32, 64, 128, 128]
    conv_x_diff_list = [0, layer_width, layer_width, layer_width, layer_width]
    conv_text_list = ['Input'] + ['Feature\nmaps'] * (len(conv_size_list) - 1)
    conv_loc_diff_list = [[3, -3]] * len(conv_size_list)

    conv_num_show_list = list(map(min, conv_num_list, [NumConvMax] * len(conv_num_list)))
    conv_top_left_list = np.c_[np.cumsum(conv_x_diff_list), np.zeros(len(conv_x_diff_list))]

    for ind in range(len(conv_size_list)):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=conv_size_list[ind],
                                    num=conv_num_list[ind],
                                    num_max=NumConvMax,
                                    num_dots=NumDots,
                                    top_left=conv_top_left_list[ind],
                                    loc_diff=conv_loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=conv_size_list[ind],
                      num=conv_num_show_list[ind],
                      top_left=conv_top_left_list[ind], loc_diff=conv_loc_diff_list[ind])
        label(conv_top_left_list[ind], conv_text_list[ind] + '\n{}@{}x{}'.format(
            conv_num_list[ind], conv_size_list[ind][0], conv_size_list[ind][1]))

    ############################
    # 卷积层之间的映射关系
    start_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    end_ratio_list = [[0.4, 0.5], [0.4, 0.8], [0.4, 0.5], [0.4, 0.8]]
    patch_size_list = [(3, 3), (2, 2), (3, 3), (2, 2)]
    ind_bgn_list = range(len(patch_size_list))
    mapping_text_list = ['Convolution', 'Max-pooling', 'Convolution', 'Max-pooling']

    for ind in range(len(patch_size_list)):
        add_mapping(
            patches, colors, start_ratio_list[ind], end_ratio_list[ind],
            patch_size_list[ind], ind,
            conv_top_left_list, conv_loc_diff_list, conv_num_show_list, conv_size_list)
        label(conv_top_left_list[ind], mapping_text_list[ind] + '\n{}x{} kernel'.format(
            patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[26, -65])

    ############################
    # 全连接层
    fc_size_list = [(fc_unit_size, fc_unit_size)] * 3
    fc_num_list = [768, 500, 2]
    fc_num_show_list = list(map(min, fc_num_list, [NumFcMax] * len(fc_num_list)))
    fc_x_diff_list = [sum(conv_x_diff_list) + layer_width, layer_width, layer_width]
    fc_top_left_list = np.c_[np.cumsum(fc_x_diff_list), np.zeros(len(fc_x_diff_list))]
    fc_loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(fc_top_left_list)
    fc_text_list = ['Hidden\nunits'] * (len(fc_size_list) - 1) + ['Outputs']

    for ind in range(len(fc_size_list)):
        if flag_omit:
            add_layer_with_omission(patches, colors, size=fc_size_list[ind],
                                    num=fc_num_list[ind],
                                    num_max=NumFcMax,
                                    num_dots=NumDots,
                                    top_left=fc_top_left_list[ind],
                                    loc_diff=fc_loc_diff_list[ind])
        else:
            add_layer(patches, colors, size=fc_size_list[ind],
                      num=fc_num_show_list[ind],
                      top_left=fc_top_left_list[ind],
                      loc_diff=fc_loc_diff_list[ind])
        label(fc_top_left_list[ind], fc_text_list[ind] + '\n{}'.format(
            fc_num_list[ind]))

    fc_text_list = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']

    for ind in range(len(fc_size_list)):
        label(fc_top_left_list[ind], fc_text_list[ind], xy_off=[-10, -65])

    ############################
    # 绘制和保存图像
    for patch, color in zip(patches, colors):
        patch.set_color(color * np.ones(3))
        if isinstance(patch, Rectangle):
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)
        elif isinstance(patch, Circle):
            patch.set_edgecolor(Black * np.ones(3))
            ax.add_patch(patch)
        elif isinstance(patch, ConnectionPatch):
            ax.add_patch(patch)

    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    plt.show()
    fig.set_size_inches(8, 2.5)

    fig_dir = './'
    fig_ext = '.png'
    fig.savefig(os.path.join(fig_dir, 'custom_convnet_fig' + fig_ext),
                bbox_inches='tight', pad_inches=0)