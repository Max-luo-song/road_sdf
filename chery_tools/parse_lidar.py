import os
import numpy as np
import argparse


def parse_lidar_pcd_file(pcd_path, return_fields=False):
    with open(pcd_path, "rb") as f:
        # 读取头部信息
        header = []
        while True:
            line = f.readline().decode("utf-8").strip()

            if line.startswith("DATA"):
                data_type = line.split()[1]
                break
            if line:
                header.append(line)

        # 解析头部信息
        def get_header_value(key):
            for h in header:
                if h.startswith(key):
                    return h.split()[1:]
            return None

        fields = get_header_value("FIELDS")
        sizes = [int(s) for s in get_header_value("SIZE")]
        types = get_header_value("TYPE")
        counts = [int(c) for c in get_header_value("COUNT")]
        points = int(get_header_value("POINTS")[0])

        # 构建 dtype
        dtype_list = []
        for i, field in enumerate(fields):
            count = counts[i]
            size = sizes[i]
            typ = types[i]

            if typ == "F":
                np_type = np.float32 if size == 4 else np.float64
            elif typ == "U":
                np_type = (
                    np.uint8
                    if size == 1
                    else np.uint16
                    if size == 2
                    else np.uint32
                    if size == 4
                    else np.uint64
                )
            elif typ == "I":
                np_type = (
                    np.int8
                    if size == 1
                    else np.int16
                    if size == 2
                    else np.int32
                    if size == 4
                    else np.int64
                )
            else:
                raise ValueError(f"不支持的类型: {typ}")

            for j in range(count):
                field_name = f"{field}_{j}" if count > 1 else field
                dtype_list.append((field_name, np_type))

        dtype = np.dtype(dtype_list)

        # 根据 data_type 读取数据
        if data_type.lower() == "ascii":
            # 读取剩余所有行
            content = f.read().decode("utf-8").strip().splitlines()
            data_list = []
            for line in content:
                parts = line.strip().split()
                if not parts:
                    continue
                # 将字符串转为 float 或 int
                data_list.append([float(p) for p in parts])
            data_np = np.array(data_list, dtype=np.float32)

            # 如果字段数不一致（部分行缺失），做下安全检查
            if data_np.shape[1] != len(dtype_list):
                raise ValueError(
                    f"数据列数 {data_np.shape[1]} 与头部字段数 {len(dtype_list)} 不匹配"
                )

            # 转成结构化数组
            structured_data = np.zeros(data_np.shape[0], dtype=dtype)
            for i, name in enumerate(dtype.names):
                structured_data[name] = data_np[:, i]
            data = structured_data

        elif data_type.lower() == "binary":
            data = np.fromfile(f, dtype=dtype, count=points)

        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        if return_fields:
            return data, fields
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 LIDAR PCD 文件转换为 CSV 文件")
    parser.add_argument("--path", type=str, help="Path to the input PCD file.")
    args = parser.parse_args()

    if args.path.endswith(".pcd"):
        parse_func = parse_lidar_pcd_file
        
    data, fields = parse_func(args.path, return_fields=True)
    
    save_path = os.path.basename(args.path).replace(".pcd", ".csv")
    with open(save_path, "w") as csv_file:
        csv_file.write(",".join(fields) + "\n")
        for point in data:
            if point.ndim == 0:
                line = ",".join(str(point[field]) for field in fields)
                csv_file.write(line + "\n")
            else:
                line = ",".join(str(value) for value in point)
                csv_file.write(line + "\n")
                
