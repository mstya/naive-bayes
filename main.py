from train import NaiveBayes
from flask import Flask
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
from flask import request

naive_bayes = NaiveBayes()
naive_bayes.train("comments.txt")

app = Flask(__name__)
api = Api(app)


class Comments(Resource):

    def post(self):
        comment = request.form.get("comment")
        result = naive_bayes.predict(comment)
        return jsonify({"result": result.item(0)})


api.add_resource(Comments, '/comments')

if __name__ == '__main__':
     app.run(port='5002')
