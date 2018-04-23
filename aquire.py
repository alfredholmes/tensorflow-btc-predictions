import csv, datetime

record = {}

with open('coinbaseUSD.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    daily_info = {}
    current_day = 0
    volume = 0
    usd_spent = 0

    open = close = None
    high = low = None

    for row in reader:
        if current_day == datetime.datetime.fromtimestamp(int(row[0])).date():
            volume = volume + float(row[2])
            usd_spent = usd_spent + float(row[2]) * float(row[1])
            close = float(row[1])
            if open is None:
                open = close
            if high is None or high < close:
                high = close
            if low is None or low > close:
                low = close
        else:
            if volume != 0:
                record[current_day] = [usd_spent / volume, volume, open, close, high, low]
                volume = 0
                usd_spent = 0
                open = None
                #print(str(current_day) + ': ' + str(usd_spent) + ' ' + str(volume))
            else:
                record[current_day] = 0
            current_day = datetime.datetime.fromtimestamp(int(row[0])).date()
            print(current_day)
with open('output.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)

    for date, data in record.items():
        #print([date])
        #print(data)
        if data != 0:
            writer.writerow([date] + data)
