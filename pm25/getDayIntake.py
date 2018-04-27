import MySQLdb
import datetime
import time

db = MySQLdb.connect("106.14.63.93", "guest", "guest", "pm25")

cursor = db.cursor()

user_name = "Annie"
dates =['2017/7/7']
for index in range(0,len(dates)):
    date = datetime.datetime.strptime(dates[index], "%Y/%m/%d")
    days_needed = 30
    begin_date = date - datetime.timedelta(days=days_needed)
    days_intake = []

    # get id
    thisID = ''
    sql = "select id from pm25.user where name = '%s'" % user_name
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            thisID = row[0]
    except:
        print "error"

    # print begin_date, thisID

    for i in range(0, days_needed + 1):
        print ((begin_date + datetime.timedelta(days=i)).strftime(
            "%Y-%m-%d"))  # , ((begin_date + datetime.timedelta(days=(i+1))).strftime("%Y-%m-%d %H:%M:%S"))
        # get one day data
        day_number = ''
        day_intake_sum = ''
        app_ver = ''
        sql2 = "select count(*), sum(pm25_intake), APP_version from pm25.data_mobile_new " \
               "where userid = '%s' and time_point > '%s' and time_point < '%s' order by time_point desc" \
               % (thisID,
                  ((begin_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")),
                  ((begin_date + datetime.timedelta(days=(i + 1))).strftime("%Y-%m-%d %H:%M:%S")))
        try:
            cursor.execute(sql2)
            results = cursor.fetchall()
            for row in results:
                day_number = row[0]
                day_intake_sum = row[1]
                app_ver = row[2]
        except:
            print "error"

        if app_ver:
            # print day_number, day_intake_sum, app_ver
            if "iOS" in app_ver:
                days_intake.append(1000 * (day_intake_sum / (day_number / float(720))))
            else:
                days_intake.append(day_intake_sum / (day_number / float(1440)))
        else:
            days_intake.append(0)

    for i in range(0, len(days_intake)):
        print days_intake[i]

db.close()
