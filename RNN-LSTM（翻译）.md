# LSTM

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 循环神经网络
人类在思考问题的时候，都是附带记忆的。当你阅读一篇文章的某个句子时，你都会带上你对前文的理解来阅读它。换句话说，你的思维是永久性的。
传统的神经网络无法做到这点，这也是传统神经网络最大的缺点。举个例子，当你试图去对一部电影中每个时刻发生的剧情进行预测，传统的神经网络是无法利用之前发生过的剧情的来预测下一秒的剧情。

递归神经网络就很好的解决了这个问题，它利用了内部的循环网络，使信息拥有了长期保留性。

<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png" height="150"/>
</p>

在上图中，输入数据Xt经过神经元模块A，得到输出Ht。信息在神经元中发生了循环的传递。这种循环模式使RNN听起来有些神秘。其实，它与普通的神经网络也差不多。我们可以想像，把一个普通神经网络复制多份，然后在这些神经网络间传递数据。就可以得到一个循环神经网络。

<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" height="150"/>
</p>

这种链式结构也暗示了RNN与序列的相关性，在处理序列类型的数据时，使用RNN更加合适。事实上，也的确如此，RNN在语音识别，语言模型，翻译，图像捕捉上都有很大作为。下面这篇来自Andrej Karpathy’s的文章对上述实现进行更详细的介绍： [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). 

RNN模型之中有个相当出色的模型叫做LSTMs，它比普通RNN模型更好，这篇文章也是重点聊聊LSTMs。

## 长期依赖(Long-Term)问题
RNN的设计思路之一就是能够将当前任务与过去的关联起来，例如在研究视频帧时，可以通过过去出现的帧来预测当前帧的信息。但RNN能否很好的做到这一点呢？依情况而定。
有些时候，我们只用考虑很近期的数据来预测当前任务的状态。例如，在一个语言模型中，我们通过前一个词来预测下一个词出现的可能性。但在这个例子中，这个关联性的距离和范围很小，仅仅是几个单词之前。RNN可以很好的使用这些过去的信息。

<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-shorttermdepdencies.png" height="150"/>
</p>

但有些时候，我们需要更远的历史信息。例如当你想预测“I grew up in France… I speak fluent French.”这句话中French语言出现的概率。如果你能拿到I grew up in France这个信息，你就可以更大概率预测出French。这时，我们需要记忆的信息更大。不幸的是，随着这个学习的“距离”越来越大，RNN的学习效果越来越差。

<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png" height="150"/>
</p>

而LSTM很好的解决了这个问题

## LSTM Networks
Long Short Term Memory networks – 也叫“LSTMs” – 是一种特殊的RNN, 它具有存储长期信息的能力. 由Hochreiter & Schmidhuber (1997)提出, 经过人们改良后，能够处理很多种类的问题, 现在广泛用于机器学习领域.

LSTMs的设计理念是，避免长期依赖问题，而是将信息看成它本身的属性。所有的RNN模型都使用链式结构，在标准的RNN模型中，重复模块里使用了一个非常简单的结构，一个单一的tanh神经网络层。
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png" height="200"/>
  
  <p align = "center"> The repeating module in a standard RNN contains a single layer.</p>
</p>

LSTMs同样也使用了链式结构，但它的重复模块使用了一种不同的结构，用了四哥网络层通过特殊的方式组合起来。

<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" height="200"/>
  
  <p align = "center"> The repeating module in an LSTM contains four interacting layers.</p>
</p>

先不用担心看不懂这些细节，接下来我们会一步一步的来介绍。首先了解下面这些符号的意义。
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png" height="150"/>
</p>

黄色的代表神经元层，粉色的表示运算，例如向量求和等。还有向量转移，合并，复制操作等等。

## LSTMs核心原理
首先我们看下上面这条线，把它想象成一个传送带。数据从Ct-1时间点沿着传送带传送到Ct时间，在这条传送带上，只有一些简单的线性运算。这时信息传递的变化不大。接着我们使用gates结构来给这条传送带上的信息进行一些调整。下面这个组合就是一个简单的gates，它由一个sigmoid神经元和一个乘运算组成。
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png" height="150"/>
</p>
当sigmoid输出为1就表示，让所有数据通过。当sigmoid输出为0就表示，所有数据都不能通过。

LSTMs有3个这种gates，下面我们一步一步来介绍。

## Step-by-Step LSTM Walk Through
第一步我们决定什么类型的信息可以流入cell state。我们管这个sigmoid层也叫做遗忘层。它根据上一时刻的输出ht-1和当前时刻的xt的权重和，使用sigmoid函数输出一个属于[0,1]范围的数字。1代表完全保留而0代表全部遗忘。

<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" height="200"/>
</p>
让我们回到一个语言模型的例子，试图根据前面的语言来预测下一个单词。在这个问题中，cell state可能包括当前的主语的词性，因此我们可以使用正确的代词来描述。但当出现了一个新的主语时，可以选择将之前的主语遗忘。

下一步是决定更新cell state中。这里有两个部分，第一，sigmoid层，我们也称之为输入gate层，它用来决定什么数据需要更新，输出为it。接着一个tanh层用来创造一个新向量Ct'。然后用这两个值相乘it * Ct'的值共同更新cell state。由于之前哦我们已经扔掉一部分信息，所以这里增加新的信息必须在扔掉原信息的基础上去做处理，所以这里依旧是用sigmoid层。
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" height="200"/>
</p>
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png" height="200"/>
</p>

最后我们决定要输出什么数据。同样我们依旧要用sigmoid把之前选择扔点的数据过滤掉，然后使用tahn（输出结果范围在[-1,1])进行调整，最终过程如下。
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" height="200"/>
</p>

## LSTM模型中的变量
迄今为止，我们谈论的是一种非常简单的LSTM模型，还有其他不同种类的LSTM模型，他们之间可能区别并不是很大，但还是很值得研究。下面是三种常见的LSTM模型。
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-peepholes.png" height="200"/>
</p>
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-tied.png" height="200"/>
</p>
<p align="center">
  <img src="http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png" height="200"/>
</p>






