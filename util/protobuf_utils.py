import os

def extract_protobuf_archive(protobuf_path):
    os.system(f"cd {protobuf_path} && tar -xf protoc-22.5-win64.zip")
    os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(protobuf_path, 'bin'))
