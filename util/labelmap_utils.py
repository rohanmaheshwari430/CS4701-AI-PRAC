def generate_label_map(labels, file_path):
    with open(file_path, 'w') as f:
        for label in labels:
            f.write('item {\n')
            f.write(f'\tname: "{label["name"]}"\n')
            f.write(f'\tid: {label["id"]}\n')
            f.write('}\n')
