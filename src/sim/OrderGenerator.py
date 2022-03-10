import random
from sim.Order import Order


class OrderGenerator:
    """ 
    provides methods to either fill the source_store with items at the start of a simulation or create a process
    that fills the given store during the simulation
    """
    """提供在模拟开始时用项目填充源存储或在模拟期间创建填充给定存储的流程的方法"""
    def __init__(self, system, order_type, order_list=None, order_probability_step=None, items_per_type=100):
        self.system = system
        
        if order_type == 'start':
            if order_list is None:
                self.generate_starting_order_alternating(items_per_type)
            else:
                self.generate_starting_order_by_list(order_list)
        elif order_type == 'process':
            assert not (order_list is None and order_probability_step is None), 'Tried to generate order_process, did not get order_list or order_probability_step.'
            assert not (order_list is not None and order_probability_step is not None), 'Tried to generate order_process, did get order_list AND order_probability_step.'
            if order_list is not None:
                self.system.sim_env.process(self.generate_order_process_list(order_list))
            if order_probability_step is not None:
                self.system.sim_env.process(self.generate_order_process_probability(order_probability_step))
        else:
            assert False, 'The given order type is not supported.'

    #初始化订单集合 根据预先设置参数
    def generate_starting_order_by_list(self, order_list):
        """ fills the source_store with all the products given in the order_list while ignoring the put_date """
        """用订单列表中给出的所有产品填充源存储，同时忽略投放日期"""
        # order_list is a list of triples (put_date, due_date, dict{product_type: amount})
        for tup in order_list:
            Order(products_to_order=tup[2], due_date=tup[1], system=self.system, order_date=0)

    #初始化订单集合 根据随机参数
    def generate_starting_order_alternating(self, items_per_type):
        """ fills the source_store with items_per_type items in alternating order with random due_dates """
        # :param items_per_type 每一类都是统一的产品数量
        """以随机到期日的交替顺序用每种类型的商品填充源商品存储"""
        for product_type in self.system.production_system.product_types:
            for _ in range(0, items_per_type):
                Order({product_type: 1}, due_date = random.randint(10, self.system.simulation_time), system = self.system, order_date = 0)
    
    def generate_order_process_probability(self, order_probability_step):
        """creates a process that runs during the simulation that puts items in the source_store at random
        (with order_probability_step {product_type: chance to be ordered in each step})"""
        """
        创建一个在模拟过程中运行的进程，该进程将项目随机放入源存储中
        （顺序概率步骤{product类型：在每个步骤产生新订单的可能性}）
        """
        while True:
            products_to_order = {}
            order_something = False
            # TODO: change this filler process to own random process of receiving orders
            # TODO: 将此填充过程更改为自己的随机接收订单过程
            # amount = random.randint(1,10)
            # if chance hits, put a product in the source_store
            # 如果运气好，就把产品放在源商店里
            for product_type in self.system.production_system.product_types:
                assert product_type in order_probability_step, 'Found a product type in the system without a probability of being ordered.'
                # TODO: could integrate putting more than one item per product_type in each step
                # TODO: 可以在每个步骤中为每个产品类型放置多个项目
                rand = random.random()
                # 如果随机数满足订单出现概率的话，订购该产品1份
                if rand < order_probability_step[product_type]:
                    products_to_order[product_type] = 1
                    order_something = True
            #如果产生新的订单，则进行下单操作
            if order_something:
                due_date = random.randint(self.system.sim_env.now, self.system.simulation_time)
                Order(products_to_order = products_to_order, due_date = due_date, system = self.system, order_date = self.system.sim_env.now)
            # 1秒后推出系统
            yield self.system.sim_env.timeout(1)
    
    def generate_order_process_list(self, order_list):
        """ creates a process that runs during the simulation that puts items in the source_store
        based on a given list of triples (put_date, due_date, {product_type: amount}) """
        # put products in order_list in the source_store at their put_date、
        """
        创建在模拟过程中运行的生产过程，将项目放入源存储中
        基于一个给定元组 (生产日期，终止日期，{产品类型：数量})
        在产品投放日期将产品投放到源商店的订单列表中
        """
        while True:
            self.system.logger.debug('Order_list: {}'.format(order_list), extra = {'simtime': self.system.sim_env.now})
            for tup in order_list:
                #到达生产时间
                if tup[0] == self.system.sim_env.now:
                    Order(products_to_order = tup[2], due_date = tup[1], system = self.system, order_date = self.system.sim_env.now)
                    #self.store.put(Product(product_type=tup[2], production_system=self.system.production_system, due_date=tup[1]))
                    self.system.logger.debug('Put order {} with due_date {}'.format(tup[2], tup[1]), extra={'simtime': self.system.sim_env.now})
                    #order_list.remove(tup)  为什么要注释？
            yield self.system.sim_env.timeout(1)
