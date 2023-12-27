# _author: wangk
# date: 2022/12/8

from flask import Flask, request, jsonify, send_file
import base64
import os
import predict
import subprocess # for voice dictation

app = Flask(__name__)

@app.route('/talk/', methods=['GET', 'POST'])
def get_human_talk():
    input_context = request.values['humanSay']
    with open('input.txt', 'w') as file1, open('input_log.txt','a') as file2:
        file1.write(input_context)
        file2.write(input_context+'\n')
    predict.main()
    with open('output.txt','r') as file3, open('output_log.txt','a') as file4:
        for line in file3:
            file4.write(line)

    # Reply with Templates
    with open('PostProcess.txt', 'r') as file5:
        lines = file5.readlines()
        input_text_t = lines[-1]  # 文件的最后一行
        input_text = input_text_t[:-1]  # 去掉最后的换行符
        input_text_list = input_text.split(' ')
        if input_text_list[0] == '1': # Learn Skill
            this_output = "尚未学习"+input_text_list[1]+"技能，现在开始教学" # I haven't learned the xx skill,you can start teaching me now.
        elif input_text_list[0] == '2': # Call Skill
            this_output = "正在执行"+input_text_list[1]+"技能"  # Executing the xx skill. 
        elif input_text_list[0] == '3': # Completion
            this_output = input_text_list[1]+"技能学习完毕"  # xx is learned.
        elif input_text_list[0] == '4': # Confirm Synonym
            this_output = input_text_list[1]
        elif input_text_list[0] == '5': # Positive Answer
            if input_text_list[1] == "对不起，我没有明白":  # Sorry, I don't understand.
                this_output = input_text_list[1]
            else:
                predict.main()
                this_output = "正在执行"+input_text_list[1]+"技能" # Executing the xx skill.
        elif input_text_list[0] == '6': # Negative Answer
            if input_text_list[1] == "对不起，我没有明白":  # Sorry, I don't understand.
                this_output = input_text_list[1]
        elif input_text_list[0] == '100':  # error
            this_output = "元技能参数不全，请查看元技能结构或咨询专业人员" # An error has occurred. Please check the primitive skill structure or consult a professional. 
            
    return this_output

@app.route('/voice/', methods=['GET', 'POST'])
def voice_dictation():
    voice_file = request.files['wavFile']
    buffer_data = voice_file.read()
    with open(os.path.join('VoiceDictation','1.wav'),'wb+') as f:
        f.write(buffer_data) # 二进制转为音频文件
    input_file = os.path.join('VoiceDictation','1.wav')
    output_file = os.path.join('VoiceDictation','2.mp3')
    # ffmpeg -y -i 1.wav -ar 16000 2.mp3
    subprocess.call(['ffmpeg','-y','-i',input_file,'-ar','16000',output_file])
    os.system('python iat_ws_python3.py')
    with open('VoiceDictation/humanWord.txt','r') as file0:
        humanInput = file0.readlines()[-1]
        # humanInput = humanInputs[-1]
        print(humanInput)
    return humanInput


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=39517) # 39517
    # app.run(debug=True, host='127.0.0.1', port=5000)
