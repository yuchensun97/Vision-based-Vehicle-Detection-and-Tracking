#%%
dets = [{'box': 4, 'id': 0},
        {'box': 3, 'id': 1},
        {'box': 8, 'id': 2},
        {'box': 6, 'id': 3}]

new_tracks = [{'boxes':[det['box']], 'start': 1} for det in dets]


# %%
class A:
        def __init__(self) :
                self.a = 100

        def update(self):
                self.a += 1
                new_A = A()
                new_A.a = self.a
                return new_A 
#%%
a_1 = A()
a_2 = a_1.update()
# %%
a = [1 for i in range(8)]
for item in a:
        a.pop(-1)
        print(item)  

a.remove(1)  

# %%
