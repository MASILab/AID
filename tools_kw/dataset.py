from tools_kw.clinical import ClinicalDataReaderSPORE
from tools_kw.utils import read_file_contents_list
from tools_kw.data_io import DataFolder


def get_data_dict(config, file_list_txt):
    in_folder = config['input_img_dir']
    label_csv = config['label_csv']

    in_folder_obj = DataFolder(in_folder, read_file_contents_list(file_list_txt))
    file_list = in_folder_obj.get_data_file_list()

    clinical_reader = ClinicalDataReaderSPORE.create_spore_data_reader_csv(label_csv)

    label_array = None
    file_list_with_valid_label = None

    # if ('two_class' in config) and (config['two_class']):
    #     label_array, file_list_with_valid_label = clinical_reader.get_label_for_CAC(file_list)
    # else:
    #     label_array, file_list_with_valid_label = clinical_reader.get_label_for_CAC_severity_level(file_list)

    label_array, file_list_with_valid_label = clinical_reader.get_label_for_CAC_3_class(file_list)

    subject_list = [ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name)
                    for file_name in file_list_with_valid_label]

    in_folder_obj.set_file_list(file_list_with_valid_label)
    file_path_list = in_folder_obj.get_file_path_list()

    data_dict = {
        'img_subs': subject_list,
        'img_files': file_path_list,
        'categories': label_array
    }

    if config['add_calcium_mask']:
        in_calcium_folder = config['input_calcium_mask']
        in_calcium_folder_obj = DataFolder(in_calcium_folder, file_list_with_valid_label)
        calcium_mask_path_list = in_calcium_folder_obj.get_file_path_list()
        data_dict['calcium_files'] = calcium_mask_path_list

    return data_dict


def get_train_test_data(config):
    train_file_list_txt = config['train_file_list_txt']
    test_file_list_txt = config['test_file_list_txt']

    train_data_dict = get_data_dict(config, train_file_list_txt)
    test_data_dict = get_data_dict(config, test_file_list_txt)

    return train_data_dict, test_data_dict
