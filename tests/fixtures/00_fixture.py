# %% [markdown]
# # Chapter 0: The Fixture
#
# A miniature chapter that puts the build through every step.
#
# > This casebook is educational content created in a personal capacity
#
# > Nothing here is investment advice or a recommendation to buy or sell any security
#
# > The data is a public research dataset with anonymised company names

# %% [markdown]
# ## The Question
#
# A chapter is written as plain text and read as a designed notebook.
# Does the build carry every part across?

# %% [markdown]
# ## What You Will Learn
#
# By the end you can say:
#
# - how a computed number reaches the prose;
# - how a chart becomes a numbered exhibit;
# - how a pitfall becomes a plate.

# %% [markdown]
# ## What You Need
#
# Nothing.
# This chapter makes its own numbers.

# %% [markdown]
# ## Your Route

# %% [markdown]
# # Setting Up
#
# Make sure the kernel `corp-default` is selected before you run a cell.
# If it is not in the list, or an import fails, the [Quick Start](../README.md#quick-start) shows how to register it and refresh the list.
#
# Q. What does every chapter do first?

# %%
import matplotlib.pyplot as plt
import numpy as np

from corp_default import store, viz

store.open_chapter(0)
viz.use_house_style()

# %% [markdown]
# **Reading.** The store is open for this chapter's numbers and the house style is set.
#
# **Next, a number travels from a cell to a sentence.**

# %% [markdown]
# # A Number in the Prose
#
# **Predict 1.** Add the whole numbers from one to six, then double the sum.
# What do you get?
#
# Q. How does a number reach a sentence?

# %%
answer = int(np.arange(1, 7).sum() * 2)
store.put("answer", answer)
answer

# %% [markdown]
# **Reading.** The values sum to {{ch0.answer}}.
# The sentence quotes the cell, so the two cannot disagree.
#
# **Check.** Where does the sentence above get its number?
#
# <details>
# <summary>Show the answer</summary>
#
# From `store.put`, which the build reads after the cell has run.
#
# </details>
#
# **Next, two charts and a plate.**

# %% [markdown]
# # Two Charts and a Plate
#
# **Predict 2.** One value in a sample of five sits far above the rest.
# Which is larger, the mean or the median?
#
# Q. What does a skewed sample look like?

# %%
sample = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
store.put("mean", sample.mean())
store.put("median", float(np.median(sample)))

fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 3.0))
ax.bar(range(1, 6), sample, color=viz.SURVIVED)
ax.set_xlabel("Observation")
ax.set_ylabel("Value")
viz.exhibit(
    fig,
    "One value towers over the rest",
    "The fifth observation sets the scale of the axis.",
    "Made up for the test",
)

# %% [markdown]
# **Reading.** Exhibit 1 shows four small values and one large one.
# The mean follows the large one.

# %% [markdown]
# <!-- pitfall: skew -->

# %% [markdown]
# Q. Does a signed scale show the small values?

# %%
fig, ax = plt.subplots(figsize=(viz.EXHIBIT_WIDTH_IN, 3.0))
ax.barh(range(1, 6), sample - sample.mean(), color=viz.FAILED)
viz.symlog_axis(ax, "x")
ax.set_xlabel("Distance from the mean")
ax.set_ylabel("Observation")
viz.exhibit(
    fig,
    "Four of five values sit below the mean",
    "A symmetric-log axis keeps both signs readable.",
    "Made up for the test",
)

# %% [markdown]
# **Reading.** Exhibit 2 puts every value on one signed scale.
#
# A drawn diagram goes in the same way as a plate.

# %% [markdown]
# <!-- diagram: label_timeline -->

# %% [markdown]
# **Check.** Why does the plate show the median beside the mean?
#
# <details>
# <summary>Show the answer</summary>
#
# The median is the typical value.
# One large observation moves the mean and leaves the median alone.
#
# </details>
#
# **The findings follow.**

# %% [markdown]
# # Findings
#
# 1. One large value sets the mean of a small sample (Exhibit 1).
# 2. A signed scale shows every value at once (Exhibit 2).

# %% [markdown]
# # Your Turn
#
# Three extensions, each with a hint folded under it.
#
# **1. A second outlier.** Add a second large value and compare the mean with the median again.
#
# <details>
# <summary>Hint</summary>
#
# Append to `sample` and rerun the cell.
#
# </details>
#
# **2. A trimmed mean.** Drop the largest value before taking the mean.
#
# <details>
# <summary>Hint</summary>
#
# Sort the sample and slice off its last value.
#
# </details>
#
# **3. A log scale.** Draw Exhibit 1 on a log axis.
#
# <details>
# <summary>Hint</summary>
#
# Call `ax.set_yscale("log")` before the exhibit is framed.
#
# </details>
