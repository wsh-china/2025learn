#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import argparse
import os
import matplotlib.font_manager as fm

# 预设的配色方案（为不同的GO类别设计）
COLOR_SCHEMES = {
    'BP': [(0, 0.5, 0.8), (0, 0.7, 1), (0.1, 0.8, 1)],  # 蓝色系
    'CC': [(0, 0.6, 0.3), (0.2, 0.8, 0.4), (0.4, 0.9, 0.5)],  # 绿色系
    'MF': [(0.7, 0.1, 0.2), (0.9, 0.2, 0.3), (1, 0.4, 0.4)],  # 红色系
    'ALL': [(0.5, 0, 0.8), (0.3, 0.3, 0.9), (0, 0.6, 0.6), (0.1, 0.8, 0.4)],  # 混合色系
    # 其他预设方案
    'blue_purple': [(0.1, 0.1, 0.9), (0.5, 0, 0.8), (0.8, 0.1, 0.5)],
    'scientific': [(0, 0.3, 0.7), (0, 0.5, 0.5), (0.1, 0.7, 0.4)],
    'elegant': [(0.2, 0.2, 0.5), (0.4, 0.2, 0.4), (0.6, 0.2, 0.3)]
}

def list_available_fonts():
    """列出系统可用的字体"""
    fonts = sorted([f.name for f in fm.fontManager.ttflist])
    return fonts

def get_font_path(font_name):
    """获取指定字体的路径"""
    if not font_name:
        return None

    # 检查是否是文件路径
    if os.path.isfile(font_name) and (font_name.endswith('.ttf') or font_name.endswith('.otf')):
        return font_name

    # 查找系统字体
    fonts = [f for f in fm.fontManager.ttflist if font_name.lower() in f.name.lower()]
    if fonts:
        return fonts[0].fname
    return None

def filter_by_ontology(df, ontology=None):
    """根据本体类别筛选数据"""
    if ontology is None or ontology.upper() == 'ALL':
        return df
    return df[df['ONTOLOGY'] == ontology.upper()]

def generate_go_wordcloud(input_file, output_prefix='go_wordcloud', use_id=False,
                         width=1200, height=800, background_color="white", max_words=1000,
                         font_name=None, color_scheme=None, random_state=42,
                         generate_all=True, generate_bp=True, generate_cc=True, generate_mf=True):
    """生成GO富集结果的词云图"""

    # 读取数据
    try:
        df = pd.read_csv(input_file)
    except:
        try:
            df = pd.read_csv(input_file, sep="\t")
        except Exception as e:
            print(f"❌ 无法读取输入文件: {e}")
            return False

    # 检查必要列
    required_cols = ['ID', 'Description', 'ONTOLOGY', 'OccurrenceCount']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 输入文件缺少必要的列: {', '.join(missing_cols)}")
        return False

    # 获取字体路径
    font_path = get_font_path(font_name)
    if font_name and font_path:
        print(f"✅ 使用字体: {font_name}")
    elif font_name:
        print(f"⚠️ 找不到字体 '{font_name}'，使用默认字体")
        font_path = None

    # 确定词云使用的文本字段
    text_field = 'ID' if use_id else 'Description'

    # 创建完整的词频字典（用于 ALL 分类）
    all_freq_dict = dict(zip(df[text_field], df['OccurrenceCount']))
    global_max_freq = max(all_freq_dict.values())  # 全局最大频率

    # 设置每个GO类别的配色和输出文件名
    ontology_settings = {
        'ALL': {'color': 'ALL', 'title': '所有GO条目词云图', 'file': f"{output_prefix}_all.png"},
        'BP': {'color': 'BP', 'title': '生物过程(BP)词云图', 'file': f"{output_prefix}_bp.png"},
        'CC': {'color': 'CC', 'title': '细胞组分(CC)词云图', 'file': f"{output_prefix}_cc.png"},
        'MF': {'color': 'MF', 'title': '分子功能(MF)词云图', 'file': f"{output_prefix}_mf.png"}
    }

    # 判断需要生成哪些词云图
    generate_flags = {
        'ALL': generate_all,
        'BP': generate_bp,
        'CC': generate_cc,
        'MF': generate_mf
    }

    # 生成各个GO类别的词云图
    for ontology, settings in ontology_settings.items():
        if not generate_flags[ontology]:
            continue

        # 根据本体类别筛选数据
        filtered_df = filter_by_ontology(df, ontology)

        if len(filtered_df) == 0:
            print(f"⚠️ 没有找到{ontology}类别的GO条目，跳过生成词云图")
            continue

        # 创建该类别的词频字典
        freq_dict = dict(zip(filtered_df[text_field], filtered_df['OccurrenceCount']))

        # 如果词典为空，跳过
        if not freq_dict:
            print(f"⚠️ {ontology}类别的词频字典为空，跳过生成词云图")
            continue

        print(f"🔍 为{ontology}类别生成词云图，包含{len(freq_dict)}个GO条目")

        # 获取特定GO类别的配色方案
        go_color_scheme = color_scheme if color_scheme else settings['color']

        # 创建颜色函数
        if go_color_scheme in COLOR_SCHEMES:
            colors = COLOR_SCHEMES[go_color_scheme]

            def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                """根据词频和预设配色方案创建颜色"""
                # 获取词的频率
                freq = freq_dict.get(word, 1)
                # 归一化频率（使用全局最大频率）
                freq_norm = min(1.0, freq / global_max_freq)

                # 根据频率在颜色列表中插值
                idx = min(int(freq_norm * (len(colors) - 1) * 100), len(colors) - 2)
                start_color = colors[idx // 100]
                end_color = colors[min(idx // 100 + 1, len(colors) - 1)]
                t = (idx % 100) / 100.0

                # 线性插值
                r = start_color[0] * (1-t) + end_color[0] * t
                g = start_color[1] * (1-t) + end_color[1] * t
                b = start_color[2] * (1-t) + end_color[2] * t

                return (int(r*255), int(g*255), int(b*255))

            color_func = custom_color_func
        else:
            color_func = None

        # 创建词云对象
        wc = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words,
            color_func=color_func,
            prefer_horizontal=0.9,
            min_font_size=10,
            max_font_size=180,
            font_step=1,
            collocations=False,
            relative_scaling=0.5,
            regexp=r"[\w\s\:\-]+",
            font_path=font_path,
            random_state=random_state
        )

        # 对于 ALL 分类，使用完整的词频字典
        if ontology == 'ALL':
            wordcloud_data = all_freq_dict
        else:
            wordcloud_data = freq_dict

        # 生成词云
        wc.generate_from_frequencies(wordcloud_data)

        # 创建图形并显示
        plt.figure(figsize=(width/100, height/100), dpi=100, facecolor=background_color)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)

        # 保存图片
        output_file = settings['file']
        try:
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor=background_color)
            print(f"✅ {ontology}词云图已保存到: {output_file}")
        except Exception as e:
            print(f"❌ 保存{ontology}词云图失败: {str(e)}")

        plt.close()

    # 显示统计信息
    bp_count = len(df[df['ONTOLOGY'] == 'BP'])
    cc_count = len(df[df['ONTOLOGY'] == 'CC'])
    mf_count = len(df[df['ONTOLOGY'] == 'MF'])

    print(f"\n📊 GO富集结果统计:")
    print(f"  • 总GO条目: {len(df)}个")
    print(f"  • 生物过程(BP): {bp_count}个")
    print(f"  • 细胞组分(CC): {cc_count}个")
    print(f"  • 分子功能(MF): {mf_count}个")

    return True

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成GO富集结果词云图')
    parser.add_argument('-i', '--input', required=True,
                        help='输入的GO富集统计CSV/TSV文件，包含ID、Description、ONTOLOGY和OccurrenceCount列')
    parser.add_argument('-o', '--output', default='go_wordcloud',
                        help='输出的词云图文件前缀 (默认: go_wordcloud)')
    parser.add_argument('-id', '--use-id', action='store_true',
                        help='使用GO ID而不是GO描述生成词云图')
    parser.add_argument('-w', '--width', type=int, default=1200,
                        help='词云图宽度 (像素, 默认: 1200)')
    parser.add_argument('-H', '--height', type=int, default=800,
                        help='词云图高度 (像素, 默认: 800)')
    parser.add_argument('-b', '--background', default='white',
                        help='背景颜色 (默认: white)')
    parser.add_argument('-n', '--max-words', type=int, default=1000,
                        help='最大显示词数 (默认: 1000)')
    parser.add_argument('-f', '--font', default=None,
                        help='指定字体名称或TTF文件路径 (默认: 系统默认字体)')
    parser.add_argument('-s', '--scheme', default=None,
                        help=f'配色方案，可选: {", ".join(COLOR_SCHEMES.keys())} (默认: 根据GO类别自动选择)')
    parser.add_argument('-r', '--random-state', type=int, default=42,
                        help='随机种子，用于控制词的位置 (默认: 42)')
    parser.add_argument('-l', '--list-fonts', action='store_true',
                        help='列出系统可用的字体')
    parser.add_argument('--no-all', action='store_true',
                        help='不生成包含所有GO类别的词云图')
    parser.add_argument('--no-bp', action='store_true',
                        help='不生成BP类别的词云图')
    parser.add_argument('--no-cc', action='store_true',
                        help='不生成CC类别的词云图')
    parser.add_argument('--no-mf', action='store_true',
                        help='不生成MF类别的词云图')

    args = parser.parse_args()

    # 列出字体
    if args.list_fonts:
        fonts = list_available_fonts()
        print("系统可用字体:")
        for i, font in enumerate(fonts, 1):
            print(f"{i}. {font}")
        return

    # 生成词云
    generate_go_wordcloud(
        args.input, args.output, args.use_id,
        args.width, args.height, args.background, args.max_words,
        args.font, args.scheme, args.random_state,
        not args.no_all, not args.no_bp, not args.no_cc, not args.no_mf
    )

if __name__ == "__main__":
    main()