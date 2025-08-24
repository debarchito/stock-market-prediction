import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""## 1. Prepare the dependencies.""")
    return


@app.cell
def _():
    import glob
    import os
    from datetime import datetime

    import jax
    import joblib
    import marimo as mo
    import matplotlib.pyplot as plt
    import optax
    import polars as pl
    import seaborn as sns
    from jax import jit, nn
    from jax import numpy as np
    from jax import random, value_and_grad

    return (
        datetime,
        glob,
        jax,
        jit,
        joblib,
        mo,
        nn,
        np,
        optax,
        os,
        pl,
        plt,
        random,
        sns,
        value_and_grad,
    )


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
    mo.md(
        r"""## 4. Define step function and parameter initialization for the LSTM model."""
    )
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
    mo.md(
        r"""## 5. Define forward pass, prediction, loss function for LSTM model and train."""
    )
    return


@app.cell
def _(
    create_sequences,
    data,
    datetime,
    jax,
    jit,
    joblib,
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
        hidden_dim = params[0].shape[0]
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

    def get_batches(key, X, y, batch_size):
        """
        Generate mini-batches of inputs and targets for training.

        Parameters:
            X (np.ndarray): Input data, shape (num_samples, seq_len, input_dim).
            y (np.ndarray): Target data, shape (num_samples,).
            batch_size (int): Number of samples per batch.

        Yields:
            Tuple[np.ndarray, np.ndarray]: Tuple of (X_batch, y_batch), where:
                - X_batch has shape (batch_size, seq_len, input_dim)
                - y_batch has shape (batch_size,)

        Notes:
            Data is shuffled at the start of each epoch.
            The last few samples are dropped if they don't fit into a full batch.
        """
        indices = random.permutation(key, len(X))
        for start_idx in range(0, len(X) - batch_size + 1, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            yield X[batch_idx], y[batch_idx]

    @jit
    def train_step(model, X_batch, y_batch, opt_state):
        loss, grads = value_and_grad(loss_fn)(model, X_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss

    seed = 42
    key = random.PRNGKey(seed)
    key, shuffle_key, lstm_key, out_key = random.split(key, 4)

    X, y = create_sequences(data)

    split_ratio = 0.8
    num_samples = len(X)
    split_index = int(num_samples * split_ratio)

    indices = random.permutation(shuffle_key, num_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    X_train, y_train = X_shuffled[:split_index], y_shuffled[:split_index]
    X_test, y_test = X_shuffled[split_index:], y_shuffled[split_index:]

    input_dim = X.shape[-1]
    hidden_dim = 64
    output_dim = 1

    params = lstm_params_init(lstm_key, input_dim, hidden_dim)
    out_w = random.normal(out_key, (output_dim, hidden_dim)) * 0.1
    out_b = np.zeros((output_dim,))

    model = (params, out_w, out_b)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(model)

    gen = 5_000
    batch_size = 128

    for epoch in range(gen):
        key, subkey = random.split(key)
        epoch_losses = []

        for X_batch, y_batch in get_batches(subkey, X_train, y_train, batch_size):
            model, opt_state, loss = train_step(model, X_batch, y_batch, opt_state)
            epoch_losses.append(loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {np.mean(np.array(epoch_losses)):.4f}")

    # Save the model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lstm_model_{timestamp}.joblib"
    joblib.dump(
        {
            "model": model,
            "opt_state": opt_state,
            "seed": seed,
            "mean": mean,
            "std": std,
        },
        model_name,
    )
    print(f"Saving model as: {model_name}")

    # Test

    params, out_w, out_b = model
    preds = jax.vmap(lambda x: predict(params, out_w, out_b, x))(X_test).squeeze()
    preds_real = preds * std[3] + mean[3]
    actual_real = y_test * std[3] + mean[3]

    # Plot

    df = pl.DataFrame(
        {
            "Index": list(range(len(X_test))),
            "Predicted": preds_real.tolist(),
            "Actual": actual_real.tolist(),
        }
    ).to_pandas()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    sns.lineplot(ax=axes[0], data=df, x="Index", y="Predicted", color="blue")
    axes[0].set_title("Predicted Close Price")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Close Price")
    axes[0].grid(True)

    sns.lineplot(ax=axes[1], data=df, x="Index", y="Actual", color="orange")
    axes[1].set_title("Actual Close Price")
    axes[1].set_xlabel("Time Step")
    axes[1].grid(True)

    sns.lineplot(
        ax=axes[2], data=df, x="Index", y="Predicted", label="Predicted", color="blue"
    )
    sns.lineplot(
        ax=axes[2], data=df, x="Index", y="Actual", label="Actual", color="orange"
    )
    axes[2].set_title("Predicted vs Actual")
    axes[2].set_xlabel("Time Step")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    return predict, split_ratio


@app.cell
def _(mo):
    mo.md(r"""## 6. Load the latest model and test.""")
    return


@app.cell
def _(
    create_sequences,
    data,
    glob,
    jax,
    joblib,
    os,
    pl,
    plt,
    predict,
    sns,
    split_ratio,
):
    joblib_files_ts = glob.glob("lstm_model_*.joblib")
    if not joblib_files_ts:
        raise FileNotFoundError("No saved joblib models found.")

    latest_file_ts = sorted(joblib_files_ts, key=os.path.getmtime)[-1]
    print(f"Loading model from: {latest_file_ts}")

    checkpoint_ts = joblib.load(latest_file_ts)

    model_ts = checkpoint_ts["model"]
    checkpoint_ts["opt_state"]
    mean_ts = checkpoint_ts["mean"]
    std_ts = checkpoint_ts["std"]
    seed_ts = checkpoint_ts["seed"]

    # Test

    X_ts, y_ts = create_sequences(data)

    key_ts = jax.random.PRNGKey(seed_ts)
    key_ts, shuffle_key_ts = jax.random.split(key_ts)

    num_samples_ts = len(X_ts)
    split_index_ts = int(split_ratio * num_samples_ts)

    indices_ts = jax.random.permutation(shuffle_key_ts, num_samples_ts)
    X_ts = X_ts[indices_ts]
    y_ts = y_ts[indices_ts]

    X_test_ts = X_ts[split_index_ts:]
    y_test_ts = y_ts[split_index_ts:]

    # Prediction

    params_ts, out_w_ts, out_b_ts = model_ts
    preds_ts = jax.vmap(lambda x: predict(params_ts, out_w_ts, out_b_ts, x))(
        X_test_ts
    ).squeeze()
    preds_real_ts = preds_ts * std_ts[3] + mean_ts[3]
    actual_real_ts = y_test_ts * std_ts[3] + mean_ts[3]

    # Plot

    df_ts = pl.DataFrame(
        {
            "Index": list(range(len(X_test_ts))),
            "Predicted": preds_real_ts.tolist(),
            "Actual": actual_real_ts.tolist(),
        }
    ).to_pandas()

    fig_ts, axes_ts = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    sns.lineplot(ax=axes_ts[0], data=df_ts, x="Index", y="Predicted", color="blue")
    axes_ts[0].set_title("Predicted Close Price")
    axes_ts[0].set_xlabel("Time Step")
    axes_ts[0].set_ylabel("Close Price")
    axes_ts[0].grid(True)

    sns.lineplot(ax=axes_ts[1], data=df_ts, x="Index", y="Actual", color="orange")
    axes_ts[1].set_title("Actual Close Price")
    axes_ts[1].set_xlabel("Time Step")
    axes_ts[1].grid(True)

    sns.lineplot(
        ax=axes_ts[2],
        data=df_ts,
        x="Index",
        y="Predicted",
        label="Predicted",
        color="blue",
    )
    sns.lineplot(
        ax=axes_ts[2], data=df_ts, x="Index", y="Actual", label="Actual", color="orange"
    )
    axes_ts[2].set_title("Predicted vs Actual")
    axes_ts[2].set_xlabel("Time Step")
    axes_ts[2].grid(True)
    axes_ts[2].legend()

    plt.tight_layout()
    plt.show()

    return


if __name__ == "__main__":
    app.run()
