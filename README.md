## Installation

```
conda create --name nl2bt python=3.6
conda activate nl2bt
pip install -r requirements.txt
```



## Previous Preparation

1. Compile and install Groot (Optional).

   It is needed if you want to visualize the behavior trees (`.xml`).

   https://github.com/BehaviorTree/Groot

2. - Download the language processing model to `flask/my_model/pytorch_model.bin`:  https://pan.baidu.com/s/1hhd-zgHZTiAlPHViVrq2xA?pwd=q4a4

     The model provided is only trained for Chinese language and our phone assembly scenario. You can use it to have a quick start.

     **OR**

   - Train your own JointBERT model for natural language processing.

     https://github.com/monologg/JointBERT

     It is needed if you are using a natural language other than Chinese and developing your own application system.

   

## Run

##### Open a new terminal:

Start the communication between the chat user interface and the natural language processing model. Wait for natural language input from the user interface, and generate processed semantic information.

```
conda activate nl2bt
cd flask && python app.py
```

##### Open a new terminal:

Start the behavior tree generation algorithm and wait for the processed semantic information.

```
conda activate nl2bt
python BTScpp/myProcess/learn_xml.py
```

##### Open a new terminal: (Optional, needed if you want to visualize the behavior trees.)

```
cd BTScpp/Groot/build && ./Groot
```

##### Open a new terminal:

Open the user interface and start inputting natural language. (If you want to use the voice input, please get your authorization (APIKey) following the tutorial of [iFLYTEK Open Platform](https://www.xfyun.cn/doc/asr/voicedictation/API.html) and add it in `flask/iat_ws_python3.py`.)

- Modify Permissions

  ```
  cd UI_linux
  chmod 777 ./chatUI.x86_64
  ```

- Run

  ```
  ./chatUI.x86_64
  ```

------

You can have a **Quick Start** by using the following Chinese input **EXAMPLES**:

Human: 请**吸取**一个软排线     （Please **pick up** a flat flexible cable）

Robot: 尚未学习**吸取**技能，现在开始教学    (I haven't learned the **pick up** skill,you can start teaching me now)

Human: 先**移动**到软排线处       (**Move to** the flat flexible cable first)

Robot: 正在执行**移动**技能       (Executing the **move to** skill)

Human: **开气泵**        (**Open the pump**)

Robot: 正在执行**开气泵**技能       (Executing the **open the pump** skill)

Human: 你已经完成了一次**吸取**        (You have already already completed a **pick up**)

Robot: **吸取**技能学习完毕    (**Pick up** is learned)

Human: 现在**关气泵**         (**Close the pump** now)

Robot: 正在执行**关气泵**技能      (Executing the **close the pump** skill)

Human: 请**抓取**一个软排线           (Please **grab** a flat flexible cable)

Robot: 检测到同义词：**抓取**和**吸取**。您是指**吸取**一个软排线吗？  (Synonymous skills detected: **grab** and **pick up**. Do you mean **pick up** a flat flexible cable?)

Human: 是的    （Yes）

Robot: 正在执行**吸取**技能      (Executing the **pick up** skill)

------

The generated tree is stored in `BTScpp/BehaviorTree.cpp/build/examples/myTree.xml`. The `BTScpp/BehaviorTree.cpp/build/examples/originalTree.xml` is the initial behavior tree library with primitive skill subtrees. Currently, its content is specific to our own assembly scenario, you can replace it with the primitive skills you need. Before each run, if you want to start from your initial library, please replace the content in `myTree.xml` with the content in `originalTree.xml`.

The `flask/my_model`, `flask/myData/intent_label.txt` and `flask/myData/slot_label.txt`  provided in our project are specifically designed and trained for Chinese natural language and our phone assembly scenario. If you are using a different natural language and developing your own system, please train your own [JointBERT]( https://github.com/monologg/JointBERT) model for natural language processing. Replace the `flask/my_model` with your trained model, and replace `flask/myData/intent_label.txt`, `flask/myData/slot_label.txt` with your training data.

Due to the policy issue, we are unable to open-source the virtual environment for our mobile phone assembly. Therefore, the provided code here only includes the part related to generating behavior trees from natural language. If you want to use the generated behavior tree (`myTree.xml`) to control your own robot, you can utilize the BehaviorTree.CPP library.  As a reference, we establish Socket communication between the behavior tree and the simulated robotic arm in Unity (Windows), using the following files

`BTScpp/BehaviorTree.CPP/sample_nodes/CMakeLists.txt`

`BTScpp/BehaviorTree.CPP/examples/Grab.cpp`

`BTScpp/BehaviorTree.CPP/sample_nodes/myTree.h`

`BTScpp/BehaviorTree.CPP/sample_nodes/myTree.cpp`

For more instructions, please refer to the official tutorial of [BehaviorTree.CPP](https://github.com/BehaviorTree/BehaviorTree.CPP/tree/v3.8).



## Note

This project was completed by our team in 2022, and the natural language processing part (slot filling, intent classification,  synonymous skill detection) can now be accomplished by large language models such as GPT-4. We highly recommend researchers to integrate large language models into the behavior tree generation algorithms to obtain more powerful language processing capabilities.