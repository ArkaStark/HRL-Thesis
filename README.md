Clone the repository:
``` git clone https://github.com/ArkaStark/HRL-Thesis.git ```

Create a python virtual environment:
```
# Create environment
python3 -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

Install the required python packages:

``` pip install -r requirements.txt  ```

Run the benchmarking script:

``` python3 benchmark.py --env="CartPole-v1" --max_episode_steps=500 --agent=a2c --train_agent ```

Check full list of arguments:

``` python3 benchmark.py --help ```
