class LinearSchedule:
    def __init__(self,
                 start,
                 finish,
                 time_length):

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.finish - self.start) / self.time_length

    def eval(self, T):
        return min(self.finish, self.start + self.delta * T)