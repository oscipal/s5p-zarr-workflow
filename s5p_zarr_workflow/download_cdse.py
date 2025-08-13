from cdsetool.query import query_features
from cdsetool.credentials import Credentials
from cdsetool.download import download_features
from cdsetool.monitor import StatusMonitor
from datetime import date
import os
import datetime
from dateutil.relativedelta import relativedelta
import tempfile
from pathlib import Path

tmp_dir = Path(tempfile.mkdtemp())
print(f"Temporary dir: {tmp_dir}")

cdse_access=os.getenv("cdse_access")
cdse_secret=os.getenv("cdse_secret")
start_date=os.getenv("start_date")
product_Type=os.getenv("productType")

startdate = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
enddate = startdate + relativedelta(days=1)

def download():
    features = query_features(
        "Sentinel5P",
        {
            "startDate": startdate,
            "completionDate": enddate,
            "productType": product_Type,
            "box": ["16","48","17","49"],
        },
    )
    download=list(
        download_features(
            features,
            tmp_dir,
            {
                "concurrency": 4,
                "monitor": StatusMonitor(),
                "credentials": Credentials(cdse_access, cdse_secret),
            },
        )
    )
    print(download)

    return tmp_dir

if __name__ == "__main__":
    print(cdse_access, start_date, enddate, product_Type)
    print(download())
