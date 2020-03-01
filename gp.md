### Gauss Process

確率を最大化するパラメータ$\bf{\theta}$を求める。

$$
k(x, x'|\theta) = \theta_{1} \exp\left(-\frac{|x-x'|^{2}}{\theta_{2}}\right) + \theta_{3} \delta(x, x') \\
\log p\left(y|X, \theta\right) \propto L = -log|K_{\theta}| - y^T K_{\theta}^{-1}y+C
$$


$$
\mathrm{argmax}_{\theta} p = \mathrm{argmax}_{\theta} L
$$
ここで、以下を用いて$L$の微分を求める。
$$
\begin{aligned}
\frac{\partial}{\partial \theta} \log|K_{\theta}| = \mathrm{Tr}\left(K_{\theta}^{-1} \frac{\partial K_{\theta}}{\partial \theta}\right) \\
\frac{\partial }{\partial \theta} K_{\theta}^{-1} = -K_{\theta}^{-1} \frac{\partial K_{\theta}}{\partial \theta} K_{\theta}^{-1}
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial \theta_{i}} &= \frac{\partial L}{\partial K_{\theta}}\frac{\partial K_{\theta}}{\partial \theta_{i}} \\
&= \sum_{n=1}^{N} \sum_{m=1}^{N} \frac{\partial L}{\partial K_{nm}}
 \frac{\partial K_{nm}}{\partial \theta_{i}} \\
 &= -\mathrm{Tr} \left(K_{\theta}^{-1} \frac{\partial K_{\theta}}{\partial \theta}\right) + \left(K_{\theta}^{-1}y\right)^{T} \frac{\partial K_{\theta}}{\partial \theta} \left(K_{\theta}^{-1}y\right)
\end{aligned}
$$
最適化のために、$\bf{\theta}\in (0, \infty]^{3}$を$\bf{\tau}= \log \bf{\theta} \in [-\infty, \infty]^{3}$に変数変換する。
$$
\begin{aligned}
\theta_{1} &= e^{\tau_{1}}, \theta_{2} = e^{\tau_{2}}, \theta_{3} = e^{\tau_{3}} \\
\tau_{1} &= \log \theta_{1}, \tau_{2} = \log \theta_{2}, \tau_{3} = \log \theta_{3}\\
k(x, x'|\tau) &= e^{\tau_{1}} \exp\left(-\frac{|x-x'|^{2}}{e^{\tau_{2}}}\right) + e^{\tau_{3}} \delta(x, x') \\
L &=  - \log|K_{\tau}| -y^{T} K_{\tau}^{-1} y + C \\
\bf{\tau}^{*} &= \argmax_{\tau} L
\end{aligned}
$$
$$
\begin{aligned}
\frac{\partial L}{\partial \tau } &= -{\rm Tr}\left(K_{\tau}^{-1} \frac{\partial K_{\tau}}{\partial \tau}\right) + (K_{\tau}^{-1} y)^{T} \frac{\partial K_{\tau}}{\partial \tau}(K_{\tau}^{-1}y)
\end{aligned}
$$

$\tau^{*}$を求めるために、目的関数を$-L$としてscipy.optimize.minimizeを用いて最適化を行う。

$$
\begin{aligned}
\frac{\partial k}{\partial \tau_{1}} &= e^{\tau_{1}} \exp \left(-\frac{|x-x'|^{2}}{e^{\tau_{2}}}\right) \\
\frac{\partial k}{\partial \tau_{2}} &=e^{\tau_{1}}  \exp \left(-\frac{|x-x'|^{2}}{e^{\tau_{2}}}\right) \frac{|x-x'|^{2}}{e^{\tau_{2}}} \\
\frac{\partial k}{\partial \tau_{3}} &= e^{\tau_{3}} \delta(x, x')
\end{aligned}
$$

参考：
$$
\begin{aligned}
\frac{\partial L}{\partial \tau} &= \sum_{i} \frac{\partial L}{\partial \theta_{i}}\frac{\partial \theta_{i}}{\partial \tau} = \frac{\partial L}{\partial \theta_{1}} e^{\tau} \\
&= \sum_{i} \frac{\partial L}{\partial K_{\theta}}\frac{\partial K_{\theta}}{\partial \theta_{i}}\frac{\partial \theta_{i}}{\partial \tau} \\
&= \frac{\partial L}{\partial K_{\theta}}\frac{\partial K_{\theta}}{\partial \theta_{1}}e^{\tau} \\
\frac{\partial L}{\partial \tau} &= \frac{\partial L}{\partial \theta_{1}}e^{\tau} = \frac{\partial L}{\partial K_{\theta}}\frac{\partial K_{\theta}}{\partial \theta_{1}}e^{\tau} \\
\frac{\partial L}{\partial \sigma} &= \frac{\partial L}{\partial \theta_{2}}e^{\sigma} = \frac{\partial L}{\partial K_{\theta}}\frac{\partial K_{\theta}}{\partial \theta_{2}}e^{\sigma} \\
\frac{\partial L}{\partial \eta} &= \frac{\partial L}{\partial \theta_{3}}e^{\eta} = \frac{\partial L}{\partial K_{\theta}}\frac{\partial K_{\theta}}{\partial \theta_{3}}e^{\eta} \\

\end{aligned}
$$
