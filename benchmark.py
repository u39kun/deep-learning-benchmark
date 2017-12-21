from importlib import import_module

frameworks = [
    'pytorch',
    # 'tensorflow'
]

models = [
    'vgg16',
    'resnet152',
    'densenet161'
]

precisions = [
    'fp32',
    'fp16'
]

class Benchmark():

    def get_framework_model(self, framework, model):
        framework_model = import_module('.'.join(['frameworks', framework, 'models']))
        return getattr(framework_model, model)

    def benchmark_model(self, mode, framework, model, precision, image_shape=(224, 224), batch_size=16, num_iterations=10):
        durations = []
        framework_model = self.get_framework_model(framework, model)(precision, image_shape, batch_size)
        for i in range(num_iterations + 1):
            duration = framework_model.eval() if mode == 'eval' else framework_model.train()
            # do not collect first run as it tends to be much slower
            if i > 0:
                durations.append(duration)
        return sum(durations) / len(durations) * 1000

    def benchmark_all(self):
        for framework in frameworks:
            for model in models:
                for precision in precisions:
                    duration = self.benchmark_model('eval', framework, model, precision)
                    print("{}'s {} eval at {}: {}ms avg".format(framework, model, precision, round(duration, 1)))
                    duration = self.benchmark_model('train', framework, model, precision)
                    print("{}'s {} train at {}: {}ms avg".format(framework, model, precision, round(duration, 1)))


if __name__ == '__main__':
    Benchmark().benchmark_all()
