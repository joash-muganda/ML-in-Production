pipeline {
    agent any
    stages {
        stage('Setup') {
            steps {
                sh 'echo "Setting up environment..."'
                sh 'pip install -r requirements.txt'  // Make sure you have a requirements.txt
            }
        }
        stage('Test') {
            steps {
                sh 'echo "Running tests"'
                sh 'pytest tests/'  // or use unittest
            }
        }
    }
    post {
        always {
            sh 'echo "Cleaning up..."'
        }
    }
}
