import os

def generate_tfrecord_script(source_dir, labelmap_file, output_file):
    script_path = os.path.join(source_dir, 'generate_tfrecord.py')
    os.system(f"python {script_path} -x {source_dir} -l {labelmap_file} -o {output_file}")
