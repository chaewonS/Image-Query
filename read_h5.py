import h5py

def print_hdf5_contents(group, prefix=""):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Group):
            print(f"Group: {prefix}/{key}")
            print_hdf5_contents(item, prefix=f"{prefix}/{key}")
        elif isinstance(item, h5py.Dataset):
            print(f"Dataset: {prefix}/{key}")
            # 데이터셋의 내용 출력 (여기서는 처음 10개의 요소만 출력)
            print(f"  Data: {item[:10]}")

# HDF5 파일 경로
file_path = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/global-feats-netvlad.h5'
# file_path = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/global-feats-superpoint-n4096-r1024.h5'

# HDF5 파일 열기 (읽기 모드로 열기)
with h5py.File(file_path, 'r') as file:
    print("File contents:")
    print_hdf5_contents(file)
