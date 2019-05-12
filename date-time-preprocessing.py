data_name = 'townsville-wavedata-1975-2019.csv'
file_open = open(data_name, 'a+')
new_data = []
file_open.seek(0)
data = file_open.readlines()
for count, line in enumerate(data):
    if count != 0:
        date_time = line.split(',')[0]
        other_data = ",".join((line.split(',')[1:]))
        if "-" in date_time:
            new_date = (date_time.split("T"))[0]
            year = new_date.split('-')[0]
            month = new_date.split('-')[1]
            day = new_date.split('-')[2]
            hour = "{}:00".format((((date_time.split("T"))[1]).split(':'))[0])
        elif "/" in date_time:
            new_date = date_time.split(" ")[0]
            year = new_date.split('/')[2]
            month = new_date.split('/')[1]
            day = new_date.split('/')[0]
            hour = "{}:00".format((date_time.split(" ")[1]).split(":")[0])
        new_line = "{},{},{},{},{}".format(year, month, day, hour, other_data)
        new_data.append(new_line)
    else:
        headers = "Year,Month,Day,Hour,Hs,HMax,Tz,Tp,Dir_Tp TRUE,SST\n"
        new_data.append(headers)

file_write = open(data_name, "w+")
file_write.writelines(new_data)
