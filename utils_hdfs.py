import hdfs
import pandas as pd
import os
import subprocess
import re

class hdfsLib:
    def __init__(self, uri='http://hadoop-hd1:50070', user='dm_fan_cx', 
                root='/user/hive/warehouse'):
        self.client = hdfs.Client(uri)
        self.user = user
        self.root = root

    # 返回这个table所有的columns
    def getTableColumns(self, table, use_table=True):
        table_name = '{}.{}'.format(self.user, table)
        if use_table:
            # 在命令行上运行这个语句，               为什么可以在后面加 readliens()
            columns = os.popen('hive -e "desc %s"' % table_name).readlines()
            columns = [col.split('\t')[0].strip() for col in columns]
            return columns


    # 从HDFS文件系统中读取文件，按行返回成list变量
    def readHDFSFile(self, file_path, delimiter='\x01'):
        lines = []
        with self.client.read(file_path) as file:
            lines = [str(line, 'utf-8').strip().split(delimiter) for line in file]
            # for line in file:
                # lines.append(str(line, 'utf-8').strip().split(delimiter))
        return lines


    # 将本地文件上的.csv文件(jupyter文件系统，即hadoop-hd5)读入，返回其pandas.DataFrame格式
    def readLocalCSV(self, table, out_dir='download', overwrite_csv=False):
        # 检测文件夹是否存在
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        table_path = '{}/{}.db/{}'.format(self.root, self.user, table)
        print('readLocalCSV : table_path =', table_path)
        csv_path   = '{}/{}.csv'.format(out_dir, table)
        if os.path.exists(csv_path):
            print('File Exists!!!')
            # 是否重写该文件
            if overwrite_csv:
                print('Remove Existing csv')
                os.remove(csv_path)
            else:
                df = pd.read_csv(csv_path)
                return df

            tmp = []
            files_list = self.client.list(table_path)
            for file in files_list:
                tmp.extend(self.readHdfsFile(os.path.join(table_path, file)))

            columns = self.getTableColumns(table)
            df = pd.DataFrame(1, columns=columns)
            df.to_csv(csv_path, index=False)
            return df


    # 从我的Hive数据库 dm_fan_cx 里面读取table，存成csv或者excel文件，存到datasource下（jupyter文件系统，即hadoop-hd5下）
    # 同时返回该table的pd.DataFrame格式
    def saveTableToLocalCSV(self, table, out_dir='datasource', overwrite_csv=False):
        # 检查该文件是否存在 
        csv_path = '{}/{}.csv'.format(out_dir, table)
        if os.path.exists(csv_path):
            print('File Exists!')
            if not overwrite_csv:
                df = pd.read_csv(csv_path)
                # 返回已经存在的csv文件
                return df
            else:
                print('Remove Existing .csv File')
                os.remove(csv_path)
        
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        table_name = '{}.{}'.format(self.user, table)
        table_path = '{}/{}.db/{}'.format(self.root, self.user, table)

        # 获取该表所有的columns名字
        columns = os.popen('hive -e "desc %s"' % table_name).readlines()
        columns = [col.split('\t')[0].strip() for col in columns]

        # 写入select出来的结果到文件中去
        cmd = ['hive', '-e', '"select * from %s"' % table_name]
        with open(csv_path, 'w') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.PIPE)

        for line in iter(process.stderr.readline, b''):
            print(line)
        process.wait()

        df = pd.read_csv(csv_path, delimiter='\t', names=columns)
        df.to_csv(csv_path, index=False)
        return df

    # 从我的Hive数据库 dm_fan_cx 里面读取table，存成本地的csv文件，存到datasource下（jupyter文件系统，即hadoop-hd5下）
    def createTableFormLocalCSV(self, table, file_path, overwrite_table=False, sep=',', encoding=None):
        
        # 检测输入文件格式是否正确 (csv格式)
        if file_path.split('.')[-1] == 'csv':
            df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        else:
            print('input file format ERROR!\n cannot read this file!')
            return None

        table_name = '{}.{}'.format(self.user, table)

        # 是否重写该table
        if overwrite_table:
            cmd = ['hive', '-e', 'DROP TABLE IF EXISTS %s' % table_name]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in iter(process.stdout.readline, b''):
                print(line)
            process.wait()


        # 将数据格式转换成我们常用的显示方式
        def chtype(s):
            s = re.sub('[\d]+', '', s)
            if s not in ['int', 'float']:
                s = 'string'
            if s == 'int':
                s = 'bigint'
            return s

        # 获取每个column的数据格式和column名
        df_type = df.dtypes.reset_index()
        df_type.columns = ['name', 'data_type']
        df_type.data_type = df_type.data_type.map(lambda s: chtype(str(s)))
        df_type['features'] = df_type.apply(lambda s: '`' + s[0] + '`' + ' ' + s[1], axis=1)
        columns = ','.join(df_type['features'])

        sql = "create table %s (%s) ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t' STORED AS TEXTFILE;" % (
            table_name, columns)
        cmd = ['hive', '-e', '"%s"' % sql]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            print(line)
        process.wait()

        for i in range(10000, 20000):
            temp_file = 'temp_save_to_hdfs_%s.csv' % i
            if not os.path.exists(temp_file):
                break
        df.to_csv(temp_file, index=False, header=False, sep='\t')

        cmd = ['hive', '-e', '"load data local inpath \'%s\' into table %s;"' % (temp_file, table_name)]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            print(line)
        process.wait()
        os.remove(temp_file)

if __name__ == '__main__':
    print('test')