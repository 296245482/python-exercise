# encoding: utf-8
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import argparse
from geopy.distance import vincenty
import sys

app = Flask(__name__)

# database config
host = '106.14.63.93'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://guest:guest@' + host + ':3306/pm25'
db = SQLAlchemy(app)


# md5 = hashlib.md5()

class Station(db.Model):
    __tablename__ = 'area_position'
    id = db.Column(db.Integer, primary_key=True)
    city = db.Column('area', db.String(80))
    name = db.Column('position_name', db.String(80))
    lat = db.Column('latitude', db.Float())
    lng = db.Column('longitude', db.Float())
    code = db.Column('station_code', db.String(10))

    def __repr__(self):
        return ("Station %s (%.5f, %.5f)" % (self.name, self.lat, self.lng)).encode('utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("lat")
    parser.add_argument("lng")
    args = parser.parse_args()
    lat = float(args.lat)
    lng = float(args.lng)
    r = 2  # search radius
    neighbors = Station.query.filter(Station.lat.between(lat - r, lat + r), Station.lng.between(lng - r, lng + r)).all()

    if len(neighbors) == 0:
        print "no near stations."
        sys.exit()
    p = (lat, lng)
    target = None
    min_dist = sys.maxint
    # find the nearest neighbor
    for x in neighbors:
        q = (x.lat, x.lng)
        dist = vincenty(p, q)
        if dist < min_dist:
            target = x
            min_dist = dist
    print target
