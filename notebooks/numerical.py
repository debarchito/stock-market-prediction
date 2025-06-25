import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## 1. Prepare the dependencies.""")
    return


@app.cell
def _():
    import jax
    import marimo as mo
    import matplotlib.pyplot as plt
    import optax
    import polars as pl
    import seaborn as sns
    from jax import nn
    from jax import numpy as np
    from jax import random, value_and_grad

    return jax, mo, nn, np, optax, pl, plt, random, sns, value_and_grad


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Prepare the data.

    We will use the [Yahoo Finance (2018–2023)](https://www.kaggle.com/datasets/suruchiarora/yahoo-finance-dataset-2018-2023) dataset, made available by [Suruchi Arora](https://www.kaggle.com/suruchiarora) on Kaggle under the [Open Data Commons Open Database License (ODbL) v1.0](http://opendatacommons.org/licenses/odbl/1.0). Additional pointers taken into consideration:

    - The **Adj Close** column is excluded, as it is identical to the **Close** column in this dataset.
    """
    )
    return


@app.cell
def _(np, pl):
    cols = ["Open", "High", "Low", "Close", "Volume"]

    data = (
        pl.read_csv("../datasets/yahoo_finance/yahoo_finance_2018_to_2023.csv")
        .drop("Adj Close")
        .sort("Date")
        .with_columns(
            [pl.col(col).str.replace_all(",", "").cast(pl.Float32) for col in cols]
        )
        .select(cols)
        .to_numpy()
        .astype(np.float32)
    )

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data - mean) / std
    return data, mean, std


@app.cell
def _(mo):
    mo.md(r"""## 3. Create input-output sequences for time series modeling.""")
    return


@app.cell
def _(np):
    def create_sequences(
        data: np.ndarray, seq_len: int = 20, target_col: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences for time series modeling.

        Parameters:
            data (np.ndarray): 2D array of shape (timesteps, features).
            seq_len (int): Number of timesteps per input sequence.
            target_col (int): Index of the target column to predict.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X: 3D array of shape (num_samples, seq_len, features)
                - y: 1D array of shape (num_samples,)
        """
        xs, ys = [], []

        for i in range(len(data) - seq_len):
            xs.append(data[i : i + seq_len])
            ys.append(data[i + seq_len][target_col])

        return np.array(xs), np.array(ys)

    return (create_sequences,)


@app.cell
def _(mo):
    mo.md(r"""## 4. Define the LSTM Cell and Parameter Initialization""")
    return


@app.cell
def _(nn, np, random):
    def lstm_step(params, carry, x):
        h, c = carry
        Wf, Wi, Wc, Wo, bf, bi, bc, bo = params
        concat = np.concatenate([h, x])
        f = nn.sigmoid(Wf @ concat + bf)
        i = nn.sigmoid(Wi @ concat + bi)
        o = nn.sigmoid(Wo @ concat + bo)
        g = np.tanh(Wc @ concat + bc)
        c = f * c + i * g
        h = o * np.tanh(c)
        return (h, c), h

    def init_lstm_params(key, input_dim, hidden_dim):
        k = input_dim + hidden_dim

        def init_gate():
            return random.normal(key, (hidden_dim, k)) * 0.1

        def init_bias():
            return np.zeros((hidden_dim,))

        return (
            init_gate(),
            init_gate(),
            init_gate(),
            init_gate(),
            init_bias(),
            init_bias(),
            init_bias(),
            init_bias(),
        )

    return


@app.cell
def _(
    create_sequences,
    data,
    jax,
    mean,
    nn,
    np,
    optax,
    pl,
    plt,
    random,
    sns,
    std,
    value_and_grad,
):
    seq_len = 20
    X, y = create_sequences(data, seq_len)
    input_dim = X.shape[-1]
    hidden_dim = 64
    output_dim = 1

    def lstm_step(params, carry, x):
        h, c = carry
        Wf, Wi, Wc, Wo, bf, bi, bc, bo = params
        concat = np.concatenate([h, x])
        f = nn.sigmoid(Wf @ concat + bf)
        i = nn.sigmoid(Wi @ concat + bi)
        o = nn.sigmoid(Wo @ concat + bo)
        g = np.tanh(Wc @ concat + bc)
        c = f * c + i * g
        h = o * np.tanh(c)
        return (h, c), h

    def init_lstm_params(key, input_dim, hidden_dim):
        k = input_dim + hidden_dim

        def init_gate():
            return random.normal(key, (hidden_dim, k)) * 0.1

        def init_bias():
            return np.zeros((hidden_dim,))

        return (
            init_gate(),
            init_gate(),
            init_gate(),
            init_gate(),
            init_bias(),
            init_bias(),
            init_bias(),
            init_bias(),
        )

    def forward(params, x):
        h = np.zeros(hidden_dim)
        c = np.zeros(hidden_dim)
        carry = (h, c)
        _, h_states = jax.lax.scan(
            lambda carry, x: lstm_step(params, carry, x), carry, x
        )
        return h_states[-1]

    def predict(params, out_w, out_b, x):
        h = forward(params, x)
        return out_w @ h + out_b

    def loss_fn(model, X, y):
        params, w, b = model
        preds = jax.vmap(lambda x: predict(params, w, b, x))(X)
        return np.mean((preds.squeeze() - y) ** 2)

    key = random.PRNGKey(0)
    params = init_lstm_params(key, input_dim, hidden_dim)
    out_w = random.normal(key, (output_dim, hidden_dim)) * 0.1
    out_b = np.zeros((output_dim,))
    model = (params, out_w, out_b)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    for epoch in range(250):
        loss, grads = value_and_grad(loss_fn)(model, X, y)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = optax.apply_updates(model, updates)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")

    params, out_w, out_b = model
    preds = jax.vmap(lambda x: predict(params, out_w, out_b, x))(X).squeeze()
    preds_real = preds * std[3] + mean[3]
    actual_real = y * std[3] + mean[3]

    plot = (
        pl.DataFrame(
            {
                "Index": list(range(250)),
                "Predicted": preds_real[-250:].tolist(),
                "Actual": actual_real[-250:].tolist(),
            }
        )
        .unpivot(
            index="Index",
            on=["Predicted", "Actual"],
            variable_name="Type",
            value_name="Price",
        )
        .to_pandas()
    )

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot, x="Index", y="Price", hue="Type")
    plt.title("LSTM Prediction vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
