import caffe

caffe.set_mode_gpu()
s = caffe.get_solver("solver.prototxt")
s.solve()
