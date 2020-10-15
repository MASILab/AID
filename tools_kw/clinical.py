import pandas as pd
import re
from datetime import datetime
from tools_kw.utils import get_logger
import numpy as np
import collections


logger = get_logger('Clinical')


class ClinicalDataReaderSPORE:
    def __init__(self, data_frame):
        self._df = data_frame

    def get_summary_characteristics_subject(self, included_subject_list):
        df_sess_list = self._df.index.to_list()
        df_subject_list = [ClinicalDataReaderSPORE._get_subject_id_from_sess_name(sess_name) for sess_name in df_sess_list]

        logger.info(f'Get the characteristics for included subjects: {len(included_subject_list)}')
        missing_subject = [subject_id for subject_id in included_subject_list if subject_id not in df_subject_list]
        if len(missing_subject) > 0:
            logger.info(f'Number of missing subject: {len(missing_subject)}')
            logger.info(missing_subject)

        included_subject_list = [subject_id for subject_id in included_subject_list if subject_id in df_subject_list]
        included_subject_idx_list = [df_subject_list.index(subject_id) for subject_id in included_subject_list]

        df_included_only = self._df.iloc[included_subject_idx_list, :]
        logger.info(f'Number rows of included only data frame: {len(df_included_only.index)}')

        # Get statics
        self._get_age_statics(df_included_only)
        self._get_bmi_statics(df_included_only)
        self._get_copd_statics(df_included_only)
        self._get_CAC_statics(df_included_only)
        self._get_race_statics(df_included_only)
        self._get_LungRADS_statics(df_included_only)
        self._get_smokingstatus_statics(df_included_only)
        self._get_packyear_statics(df_included_only)
        self._get_education_statics(df_included_only)
        self._get_cancer_statics(df_included_only)
        self._get_plco_statics(df_included_only)

    def get_CAC_statics_summary(self, file_list):
        """
        TODO
        Get the statistic on both scan and subject level.
        :param file_list:
        :return:
        """
        pass

    def save_csv(self, out_csv):
        self._df.to_csv(out_csv)

    def _get_age_statics(self, df):
        column_str = 'age'
        ranges = [0, 55, 60, 65, 70, 75,100]
        self._get_continue_column_statics(df, column_str, ranges)

    def _get_bmi_statics(self, df):
        column_str = 'bmi'
        ranges = [0, 18.5, 24.9, 30.0, 100]
        self._get_continue_column_statics(df, column_str, ranges)

    def _get_copd_statics(self, df):
        column_str = 'copd'
        self._get_discrete_column_statics(df, column_str)

    def _get_CAC_statics(self, df):
        column_str = 'Coronary Artery Calcification'
        self._get_discrete_column_statics(df, column_str)

    def _get_race_statics(self, df):
        column_str = 'race'
        self._get_discrete_column_statics(df, column_str)

    def _get_LungRADS_statics(self, df):
        column_str = 'LungRADS'
        self._get_discrete_column_statics(df, column_str)

    def _get_smokingstatus_statics(self, df):
        column_str = 'smokingstatus'
        self._get_discrete_column_statics(df, column_str)

    def _get_packyear_statics(self, df):
        column_str = 'packyearsreported'
        ranges = [0, 30, 60, 90, 500]
        self._get_continue_column_statics(df, column_str, ranges)

    def _get_education_statics(self, df):
        column_str = 'education'
        self._get_discrete_column_statics(df, column_str)
    
    def _get_cancer_statics(self, df):
        column_str = 'cancer_bengin'
        self._get_discrete_column_statics(df, column_str)

    def _get_plco_statics(self, df):
        column_str = 'plco'
        ranges = [0, 100]
        self._get_continue_column_statics(df, column_str, ranges)

    def _get_continue_column_statics(self, df, column_str, ranges):
        sample_size = len(df.index)
        num_missing = df[column_str].isnull().sum()
        value_bins_count = df[column_str].value_counts(bins=ranges, sort=False)
        value_bins_percentage = value_bins_count * 100 / sample_size
        count_df = pd.DataFrame({'Count': value_bins_count, '%': value_bins_percentage})
        missing_row = pd.Series(data={'Count': num_missing, '%': num_missing * 100 / sample_size}, name='Missing')
        count_df.append(missing_row, ignore_index=False)
        logger.info('')
        logger.info(f'Statics {column_str}')
        print(count_df)
        print(f'Missing: {num_missing} ({num_missing * 100 / sample_size} %)')

    def _get_discrete_column_statics(self, df, column_str):
        sample_size = len(df.index)
        num_missing = df[column_str].isnull().sum()
        value_count = df[column_str].value_counts()
        value_percentage = value_count * 100 / sample_size
        count_df = pd.DataFrame({'Count': value_count, '%': value_percentage})
        missing_row = pd.Series(data={'Count': num_missing, '%': num_missing * 100 / sample_size}, name='Missing')
        count_df.append(missing_row, ignore_index=False)
        logger.info('')
        logger.info(f'Statics {column_str}')
        print(count_df)
        print(f'Missing: {num_missing} ({num_missing * 100 / sample_size} %)')

    def get_attributes_from_original_label_file(self, df_ori, attribute):
        logger.info(f'Add attribute {attribute} from ori df')
        attribute_val_list = []

        for sess, item in self._df.iterrows():
            subject_id = ClinicalDataReaderSPORE._get_subject_id_from_sess_name(sess)
            name_field_flat = ClinicalDataReaderSPORE._get_name_field_flat_from_sub_id(subject_id)
            date_obj = ClinicalDataReaderSPORE._get_date_obj_from_sess_name(sess)

            # print(df_ori['studydate'])
            # print(np.datetime64(date_obj))
            sess_row = df_ori[(df_ori['SPORE'] == name_field_flat) & (df_ori['studydate'] == np.datetime64(date_obj))]

            if sess_row.shape[0] == 0:
                # logger.info(f'Cannot find label item for {sess}')
                nearest_scan = self.check_nearest_record_for_impute(df_ori, sess + '.nii.gz')
                nearest_sess = nearest_scan.replace('.nii.gz', '')
                date_obj = ClinicalDataReaderSPORE._get_date_obj_from_sess_name(nearest_sess)
                sess_row = df_ori[(df_ori['SPORE'] == name_field_flat) & (df_ori['studydate'] == np.datetime64(date_obj))]

            if sess_row.shape[0] > 0:
                attribute_val = sess_row.iloc[0][attribute]
                attribute_val_list.append(attribute_val)
            else:
                logger.info(f'Cannot find label item for {sess}')
                attribute_val_list.append(np.nan)

        if attribute == 'packyearsreported':
            for idx_val in range(len(attribute_val_list)):
                if attribute_val_list[idx_val] >500:
                    attribute_val_list[idx_val] = np.nan

        self._df[attribute] = attribute_val_list

    def filter_sublist_with_label(self, in_file_list, field_name, field_flag):
        print(f'Filtering file list with field_name ({field_name}) and flag ({field_flag})', flush=True)
        out_file_list = []
        for file_name in in_file_list:
            subject_id = self._get_subject_id_from_file_name(file_name)
            if self._if_field_match(subject_id, field_name, field_flag):
                out_file_list.append(file_name)

        print(f'Complete. Find {len(out_file_list)} matching items.')
        return out_file_list

    def _if_field_match(self, subj, field_name, field_flag):
        subj_name = self._get_name_field_flat_from_sub_id(subj)
        row_id_list = self._df.loc[self._df['sub_name'] == subj_name]

        if_match = False
        if len(row_id_list) == 0:
            print(f'Cannot find subject id {subj_name}', flush=True)
        else:
            row_id = self._df.loc[self._df['sub_name'] == subj_name].index[0]
            sex_str = str(self._df.at[row_id, field_name])
            # sex_str = str(self._df.get_value(row_id, field_name))
            if_match = sex_str == field_flag

        return if_match

    def check_if_have_record(self, file_name_nii_gz):
        spore_name_field = self._get_name_field_flat_from_sub_id(
            self._get_subject_id_from_file_name(file_name_nii_gz)
        )
        spore_date_field = self._get_date_str_from_file_name(file_name_nii_gz)
        filtered_rows = \
            self._df[(self._df['SPORE'] == spore_name_field) & (self._df['studydate'] == np.datetime64(spore_date_field))]

        count_row = filtered_rows.shape[0]

        return count_row != 0

    def check_nearest_record_for_impute(self, df, file_name_nii_gz):
        cp_record_file_name = None

        spore_name_field = self._get_name_field_flat_from_sub_id(
            self._get_subject_id_from_file_name(file_name_nii_gz)
        )
        spore_date_field = self._get_date_str_from_file_name(file_name_nii_gz)
        spore_date_field = np.datetime64(spore_date_field)
        filtered_rows_name_field = \
            df[(df['SPORE'] == spore_name_field)]

        if filtered_rows_name_field.shape[0] > 0:
            time_list = filtered_rows_name_field['studydate'].to_numpy()
            time_delta_list = time_list - spore_date_field
            time_days_list = time_delta_list / np.timedelta64(1, 'D')
            time_days_list = np.abs(time_days_list)
            min_idx = np.argmin(time_days_list)

            subj_id = self._get_subject_id_from_file_name(file_name_nii_gz)
            time_closest = time_list[min_idx]
            datetime_obj = pd.to_datetime(str(time_closest))
            datetime_str = datetime_obj.strftime('%Y%m%d')
            cp_record_file_name = f'{subj_id:08}time{datetime_str}.nii.gz'

        return cp_record_file_name

    def get_value_field(self, file_name_nii_gz, field_flag):
        spore_name_field = self._get_name_field_flat_from_sub_id(
            self._get_subject_id_from_file_name(file_name_nii_gz)
        )
        spore_date_field = self._get_date_str_from_file_name(file_name_nii_gz)

        filtered_rows = \
            self._df[(self._df['SPORE'] == spore_name_field) & (self._df['studydate'] == np.datetime64(spore_date_field))]

        return_val = np.nan
        count_row = filtered_rows.shape[0]
        if count_row == 0:
            logger.info(f'Cannot find label item for {file_name_nii_gz}')
        else:
            return_val = filtered_rows.iloc[0][field_flag]
            # logger.info(f'Field value: {return_val}')

        return return_val

    def is_first_cancer_scan(self, file_name_nii_gz):
        spore_name_field = self._get_name_field_flat_from_sub_id(
            self._get_subject_id_from_file_name(file_name_nii_gz)
        )
        spore_date_field = self._get_date_str_from_file_name(file_name_nii_gz)
        spore_date_field = np.datetime64(spore_date_field.strftime('%Y-%m-%d'))

        subject_df = self._df[self._df['SPORE'] == spore_name_field]
        study_date_list = subject_df['studydate'].values
        # print(study_date_list)
        # print(spore_date_field)
        # print(study_date_list - spore_date_field)
        study_date_delta_days = (study_date_list - spore_date_field) / \
                                np.timedelta64(1, 'D')
        min_delta = np.min(study_date_delta_days)
        return int(min_delta == 0)

    def get_binary_classification_label(self, attr_flag, file_list):
        label_array = []
        valid_file_list = []
        if attr_flag == 'obese':
            label_array, valid_file_list = self.get_label_for_obese(file_list)
        elif attr_flag == 'COPD':
            label_array, valid_file_list = self.get_label_for_COPD(file_list)
        elif attr_flag == 'CAC':
            label_array, valid_file_list = self.get_label_for_CAC(file_list)
        else:
            raise NotImplementedError

        # change to calculate the number of subject instead of scans
        valid_subject_list = [self._get_subject_id_from_file_name(file_name) for file_name in valid_file_list]
        print(len(valid_subject_list))
        unique_subject_list = list(set(valid_subject_list))
        subject_label_list = [label_array[valid_subject_list.index(subject_id)] for subject_id in unique_subject_list]

        num_subject_total = len(unique_subject_list)
        num_subject_pos = np.sum(np.array(subject_label_list))
        num_subject_neg = num_subject_total - num_subject_pos
        print(f'Number of pos subject: {num_subject_pos} ({100 * num_subject_pos / num_subject_total:.2f}%)')
        print(f'Number of pos subject: {num_subject_neg} ({100 * num_subject_neg / num_subject_total:.2f}%)')

        num_pos = np.sum(label_array)
        num_total = len(label_array)
        num_neg = num_total - num_pos
        print(f'Number of pos cases: {num_pos} ({100 * num_pos / num_total:.2f}%)')
        print(f'Number of neg cases: {num_neg} ({100 * num_neg / num_total:.2f}%)')

        return label_array, valid_file_list

    def get_label_for_obese(self, file_list):
        logger.info(f'Get label for obese')

        obese_threshold = 30

        file_list_without_ext = [file_name.replace('.nii.gz', '') for file_name in file_list]
        used_df = self._df.loc[file_list_without_ext, ['bmi']]
        # df_scan_list = used_df.index.to_list()
        # print([item for item, count in collections.Counter(df_scan_list).items() if count > 1])
        # print(len(df_scan_list))
        # print(len(list(set(df_scan_list))))
        bmi_array = np.array(used_df['bmi'].to_list())
        # print(len(bmi_array))
        # valid_idx_list = np.argwhere((~np.isnan(bmi_array)) and ((bmi_array > 30) or (bmi_array > 18.5 and bmi_array < 24.9)))[:, 0]
        valid_idx_list = np.argwhere(~np.isnan(bmi_array))[:, 0]
        file_list_with_valid_label = [file_list[idx] for idx in valid_idx_list]
        logger.info(f'Number of scans without valid label: {len(file_list) - len(valid_idx_list)}')
        # print(bmi_array.shape)
        bmi_array = bmi_array[valid_idx_list]

        obese_label = np.zeros((len(bmi_array),), dtype=int)
        for idx in range(len(obese_label)):
            bmi = bmi_array[idx]
            if bmi >= obese_threshold:
                obese_label[idx] = 1
            elif (bmi > 18.5) & (bmi < 24.9):
                obese_label[idx] = 2
        valid_idx_list = np.argwhere(obese_label != 0)[:, 0]
        bmi_array = bmi_array[valid_idx_list]

        obese_label = np.zeros((len(bmi_array),), dtype=int)
        for idx in range(len(obese_label)):
            bmi = bmi_array[idx]
            if bmi > obese_threshold:
                obese_label[idx] = 1

        file_list_with_valid_label = [file_list_with_valid_label[idx] for idx in valid_idx_list]

        # print(obese_label.shape)
        # print(len(file_list_with_valid_label))

        return obese_label, file_list_with_valid_label

    def get_label_for_COPD(self, file_list):
        logger.info(f'Get label for COPD')

        file_list_without_ext = [file_name.replace('.nii.gz', '') for file_name in file_list]
        used_df = self._df.loc[file_list_without_ext, ['copd']]

        copd_array = np.array(used_df['copd'].to_list())
        valid_idx_list = [idx for idx in range(len(copd_array)) if str(copd_array[idx]) != 'nan']
        copd_array = [copd_array[idx] for idx in valid_idx_list]
        copd_label = np.zeros((len(copd_array),), dtype=int)

        for idx_sample in range(len(copd_array)):
            if copd_array[idx_sample] == 'Yes':
                copd_label[idx_sample] = 1
            elif copd_array[idx_sample] == 'No':
                copd_label[idx_sample] = 0
            else:
                raise NotImplementedError

        logger.info(f'Number of scans without valid label: {len(file_list) - len(copd_array)}')
        file_list_with_valid_label = [file_list[idx] for idx in valid_idx_list]

        return copd_label, file_list_with_valid_label

    def get_label_for_CAC_severity_level(self, file_list):
        """
        Instead of the dichotomous labeling for CAC, this function return the CAC severity level (4).
        :param file_list:
        :return:
        """
        logger.info(f'Get label for CAC')

        label_array = []
        file_list_with_valid_label = []

        file_list_without_ext = [file_name.replace('.nii.gz', '') for file_name in file_list]
        used_df = self._df.reindex(file_list_without_ext)

        # used_df = self._df.loc[file_list_without_ext, ['Coronary Artery Calcification']]

        cac_array = used_df['Coronary Artery Calcification'].to_list()
        valid_idx_list = [idx for idx in range(len(cac_array)) if str(cac_array[idx]) != 'nan']

        cac_array = [cac_array[idx] for idx in valid_idx_list]
        label_array = np.zeros((len(cac_array),), dtype=int)

        CAC_label_mapping = {
            'None': 0,
            'Mild': 1,
            'Moderate': 2,
            'Severe':3
        }

        for idx_sample in range(len(cac_array)):
            label_array[idx_sample] = CAC_label_mapping[cac_array[idx_sample]]

        logger.info(f'Number of invalid labels: {len(file_list) - len(cac_array)}')
        file_list_with_valid_label = [file_list[idx] for idx in valid_idx_list]

        return label_array, file_list_with_valid_label

    def get_label_for_CAC_3_class(self, file_list):
        """
        Instead of the dichotomous labeling for CAC, this function return the CAC severity level (4).
        :param file_list:
        :return:
        """
        logger.info(f'Get label for CAC')

        label_array = []
        file_list_with_valid_label = []

        file_list_without_ext = [file_name.replace('.nii.gz', '') for file_name in file_list]
        used_df = self._df.reindex(file_list_without_ext)

        # used_df = self._df.loc[file_list_without_ext, ['Coronary Artery Calcification']]

        cac_array = used_df['Coronary Artery Calcification'].to_list()
        valid_idx_list = [idx for idx in range(len(cac_array)) if str(cac_array[idx]) != 'nan']

        cac_array = [cac_array[idx] for idx in valid_idx_list]
        label_array = np.zeros((len(cac_array),), dtype=int)

        CAC_label_mapping = {
            'None': 0,
            'Mild': 0,
            'Moderate': 1,
            'Severe': 2
        }

        for idx_sample in range(len(cac_array)):
            label_array[idx_sample] = CAC_label_mapping[cac_array[idx_sample]]

        logger.info(f'Number of invalid labels: {len(file_list) - len(cac_array)}')
        file_list_with_valid_label = [file_list[idx] for idx in valid_idx_list]

        return label_array, file_list_with_valid_label

    def get_label_for_CAC(self, file_list):
        logger.info(f'Get label for CAC')

        file_list_without_ext = [file_name.replace('.nii.gz', '') for file_name in file_list]
        used_df = self._df.loc[file_list_without_ext, ['Coronary Artery Calcification']]

        cac_array = used_df['Coronary Artery Calcification'].to_list()
        valid_idx_list = [idx for idx in range(len(cac_array)) if str(cac_array[idx]) != 'nan']

        cac_array = [cac_array[idx] for idx in valid_idx_list]
        label_array = np.zeros((len(cac_array),), dtype=int)

        for idx_sample in range(len(cac_array)):
            if (cac_array[idx_sample] == 'Moderate') | (cac_array[idx_sample] == 'Severe'):
                label_array[idx_sample] = 1

        logger.info(f'Number of invalid labels: {len(file_list) - len(cac_array)}')
        file_list_with_valid_label = [file_list[idx] for idx in valid_idx_list]

        return label_array, file_list_with_valid_label

    def get_conditioned_file_list_CAC(self, file_list, value_str):
        logger.info(f'Get label for CAC')

        file_list_without_ext = [file_name.replace('.nii.gz', '') for file_name in file_list]
        used_df = self._df.loc[file_list_without_ext, ['Coronary Artery Calcification']]

        conditioned_df = used_df[used_df['Coronary Artery Calcification'] == value_str]

        conditioned_file_list_wo_ext = conditioned_df.index.to_list()
        conditioned_file_list = [file_name_wo_ext + '.nii.gz' for file_name_wo_ext in conditioned_file_list_wo_ext]

        conditioned_file_list = list(set(conditioned_file_list))

        return conditioned_file_list

    @staticmethod
    def _get_date_str_from_file_name(file_name_nii_gz):
        match_list = re.match(r"(?P<subject_id>\d+)time(?P<time_id>\d+).nii.gz", file_name_nii_gz)
        date_str_ori = match_list.group('time_id')
        date_obj = datetime.strptime(date_str_ori, "%Y%m%d")
        # date_str = date_obj.strftime("%m/%d/%Y")
        return date_obj

    @staticmethod
    def _get_date_obj_from_sess_name(sess_name):
        match_list = re.match(r"(?P<subject_id>\d+)time(?P<time_id>\d+)", sess_name)
        date_str_ori = match_list.group('time_id')
        date_obj = datetime.strptime(date_str_ori, "%Y%m%d")
        return date_obj

    @staticmethod
    def _get_subject_id_from_file_name(file_name_nii_gz):
        match_list = re.match(r"(?P<subject_id>\d+)time(?P<time_id>\d+).nii.gz", file_name_nii_gz)
        subject_id = int(match_list.group('subject_id'))
        return subject_id

    @staticmethod
    def _get_subject_id_from_sess_name(sess_list):
        match_list = re.match(r"(?P<subject_id>\d+)time(?P<time_id>\d+)", sess_list)
        subject_id = int(match_list.group('subject_id'))
        return subject_id

    @staticmethod
    def _get_name_field_flat_from_sub_id(sub_id):
        return f'SPORE_{sub_id:08}'

    @staticmethod
    def create_spore_data_reader_xlsx(file_xlsx):
        return ClinicalDataReaderSPORE(pd.read_excel(file_xlsx))

    @staticmethod
    def create_spore_data_reader_csv(file_csv):
        return ClinicalDataReaderSPORE(pd.read_csv(file_csv, index_col='id'))

    @staticmethod
    def get_subject_list(file_list):
        subject_id_list = list(set([ClinicalDataReaderSPORE._get_subject_id_from_file_name(file_name) for file_name in file_list]))
        return subject_id_list
    