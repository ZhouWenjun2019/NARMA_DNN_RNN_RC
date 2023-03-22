NARMA10 task

$$y(t+1)=0.3 y(t)+0.05 y(t)\left(\sum_{i=1}^9 y(t-i)\right)+1.5 u(t-9) u(t)+0.1$$

Results:

- RC: ~0.15

- DNN

  | Input nodes | NMSE      |
  | ----------- | --------- |
  | 10          | 0.201     |
  | **20**      | **0.023** |
  | 15          | 0.036     |
  | 11          | 0.093     |

- RNN

  | Time steps | NMSE       |
  | ---------- | ---------- |
  | 10         | 0.0731     |
  | 20         | 0.072      |
  | 15         | 0.0759     |
  | **11**     | **0.0612** |

Something interesting:

- In RNN/DNN, it's better to set the output size 1;
- In RNN, it's better to keep the hidden states to the next sample in each phase;
- In DNN, it's better to set the input dim bigger than 10, otherwise, the performance will be worse;
- Usually, the performance of RC is worse than RNN/DNN;
- In my script, the performance of RNN is worse than DNN, and I didn't look into it.