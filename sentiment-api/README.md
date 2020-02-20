# Thai Sentiment

**How to start**

```bash
# start venv
python3 -m venv venv

# Enter python virtual environment
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

**Training Model**

```bash
python train.py
```

**Start Web server**

```bash
python main.py
```

**Docker**

```bash
# build docker image
docker build -t sentiment .

# run docker container
docker run -it --rm -p 0.0.0.0:8000:8000 sentiment
```

Test
[localhost:8000/api/analyze?text=วันนี้อากาศดีมากๆเลย](http://localhost:8000/api/analyze?text=วันนี้อากาศดีมากๆเลย)
