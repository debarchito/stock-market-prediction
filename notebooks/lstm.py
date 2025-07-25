import marimo

__generated_with = "0.14.13"
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
    mo.md(r"""## 4. Define step function and parameter initialization for the LSTM model.""")
    return


@app.cell
def _(nn, np, random):
    def lstm_step(
        params: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        carry: tuple[np.ndarray, np.ndarray],
        x: np.ndarray,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Perform a single LSTM step.

        Parameters:
            params (tuple): LSTM parameters (Wf, Wi, Wc, Wo, bf, bi, bc, bo).
            carry (tuple): Previous hidden state (h) and cell state (c).
            x (np.ndarray): Input at the current time step.

        Returns:
            tuple: Updated carry (h, c) and the new hidden state h.
        """
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

    def lstm_params_init(
        key: int, input_dim: int, hidden_dim: int
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Initialize LSTM weights and biases.

        Parameters:
            keys: Random seed
            input_dim: Size of input vector
            hidden_dim: Size of hidden state

        Returns:
            tuple of weights and biases for each gate
        """
        k = input_dim + hidden_dim
        [k1, k2, k3, k4] = random.split(key, 4)

        return (
            random.normal(k1, (hidden_dim, k)) * 0.1,
            random.normal(k2, (hidden_dim, k)) * 0.1,
            random.normal(k3, (hidden_dim, k)) * 0.1,
            random.normal(k4, (hidden_dim, k)) * 0.1,
            np.zeros((hidden_dim,)),
            np.zeros((hidden_dim,)),
            np.zeros((hidden_dim,)),
            np.zeros((hidden_dim,)),
        )

    return lstm_params_init, lstm_step


@app.cell
def _(mo):
    mo.md(r"""## 5. Define forward pass, prediction, loss function for LSTM model and train.""")
    return


@app.cell
def _(
    create_sequences,
    data,
    jax,
    lstm_params_init,
    lstm_step,
    mean,
    np,
    optax,
    pl,
    plt,
    random,
    sns,
    std,
    value_and_grad,
):
    def forward(
        params: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Run the LSTM forward over a sequence of inputs.

        Parameters:
            params: tuple of LSTM parameters (weights and biases).
            x: Input sequence, shape (seq_len, input_dim).

        Returns:
            The final hidden state after processing the sequence, shape (hidden_dim,).
        """
        carry = (np.zeros(hidden_dim), np.zeros(hidden_dim))
        _, h_states = jax.lax.scan(
            lambda carry, x: lstm_step(params, carry, x), carry, x
        )

        return h_states[-1]

    def predict(
        params: tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
        out_w: np.ndarray,
        out_b: np.ndarray,
        x: np.ndarray,
    ) -> np.ndarray:
        """
        Predict the output for a given input sequence using the LSTM and output layer.

        Parameters:
            params: LSTM parameters.
            out_w: Output layer weight matrix.
            out_b: Output layer bias vector.
            x: Input sequence, shape (seq_len, input_dim).

        Returns:
            Predicted output vector.
        """
        return out_w @ forward(params, x) + out_b

    def loss_fn(
        model: tuple[
            tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            np.ndarray,
            np.ndarray,
        ],
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Compute mean squared error loss over a batch of input-output pairs.

        Parameters:
            model: tuple containing (LSTM params, output weights, output bias).
            X: Batch of input sequences, shape (batch_size, seq_len, input_dim).
            y: True target values, shape (batch_size,).

        Returns:
            Scalar mean squared error loss.
        """
        params, w, b = model
        preds = jax.vmap(lambda x: predict(params, w, b, x))(X)

        return np.mean((preds.squeeze() - y) ** 2)

    # Train

    key = random.PRNGKey(0)

    X, y = create_sequences(data)
    input_dim = X.shape[-1]
    hidden_dim = 64
    output_dim = 1

    params = lstm_params_init(key, input_dim, hidden_dim)

    out_w = random.normal(key, (output_dim, hidden_dim)) * 0.1
    out_b = np.zeros((output_dim,))

    model = (params, out_w, out_b)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    gen = 251

    for epoch in range(gen):
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
                "Index": list(range(gen)),
                "Predicted": preds_real[-gen:].tolist(),
                "Actual": actual_real[-gen:].tolist(),
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
