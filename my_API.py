from flask import Flask, render_template, request
from flask_cors import CORS
import os
import my_YoloV8
import cv2
import random
import imghdr
# import magic
# import string

# from random import random
# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"
model = my_YoloV8.YOLOv8_ObjectCounter()

# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
        try:
            # Lấy file gửi lên
            file = request.files['file']
            # print(image.headers,image.filename)
            if file:
                # Lưu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                print("Save = ", path_to_save)
                file.save(path_to_save)
                colors = []
                for _ in range(80):
                    rand_tuple = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                    colors.append(rand_tuple)
                check_image = imghdr.what(path_to_save)
                if check_image:

                    # Convert image to dest size tensor
                    frame = cv2.imread(path_to_save)

                    results = model.predict_img(frame)
                    result_img = model.custom_display(colors=colors)
                    if len(results) > 0:

                        dictObject, save_name = model.count_object(results, app.config['UPLOAD_FOLDER'], result_img)
                        # Trả về kết quả
                        return render_template("index.html", user_image="/yolov8/" + save_name,
                                               msg="Tải file lên thành công", name_Object=dictObject)
                    else:
                        return render_template('index.html', msg='Không nhận diện được vật thể')
                else:
                    totalCount,dictObject,save_file = model.predict_video(video_path=path_to_save,
                                                  save_dir =app.config['UPLOAD_FOLDER']+"/video/",
                                                  save_format = "avi",
                                                  display = 'custom',
                                                  colors = colors)
                    print(totalCount,dictObject,save_file)
                    video = model.convert_video(save_file, app.config['UPLOAD_FOLDER']+"/videoOut/")
                    # print(video)
                    if totalCount > 0:
                        # Trả về kết quả
                        return render_template("index.html",
                                               msg="Tải file lên thành công", name_Object=dictObject, video=video)
                    else:
                        return render_template('index.html', msg='Không nhận diện được vật thể')
            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Hãy chọn file để tải lên')

        except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Không nhận diện được vật thể')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
