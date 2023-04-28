import argparse

import pickle
import time

import math
import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib as mpl

import copy

from IPython.display import display, clear_output, Latex

from ipywidgets import interact, interactive, fixed, interact_manual, HBox, Layout,VBox
import ipywidgets as widgets

import itertools
import sympy as sp


##%matplotlib inline


b = 4
num_middle_node = 0
dumbbell_type = 'normal'

cd_min = 1  ## the smallest cardinality to start with

x, y = sp.symbols("x y")


def nodes_pos_polygon(b, d=None, offset=[0,0], start_node=0, last_radian=0):
    """d is radius of the circle the polygon nodes in"""
    if d == None:
        d = b
    pos = dict()
    for i in range(0, b):
        pos[i+start_node] =  (round(offset[0]+d * np.cos((i+1)/b*2*np.pi + last_radian),8), 
                  round(offset[1]+d * np.sin((i+1)/b*2*np.pi + last_radian),8))
    # print('polygon pos', pos)
    return pos

def nodes_pos_connecting(b, d=None, offset=[0,0], num_middle_node=0, start_node=None, start_pos=None):
    """d is radius of the circle the polygon nodes in"""
    if num_middle_node==0 or num_middle_node==-1:
        return
    if d == None:
        d = b
    pos = dict()
    if start_node == None:
        start_node=b
    for i in range(0, num_middle_node):
        pos[i+start_node] = (offset[0]+ d * (i+2), offset[1])
    return pos


def edges_clique(b, start_node=0):
    e_list = []
    for i in range(start_node, b+start_node):
        for j in range(i+1, b+start_node):
            e_list.append((i, j, dict(weight = 1)))
    return e_list

def edges_polygon(b, start_node=0):
    e_list = []
    for i in range(start_node, b+start_node):
        e_list.append((i, i+1 - b*((i+1)//(b+start_node)), dict(weight = 1)))
        # print(i, e_list)
    return e_list

def edges_connecting(b, start_node=None, num_middle_node=0):
    if num_middle_node==-1:
        return []
    if start_node==None:
        start_node=b
    e_list = []
    for i in range(0, num_middle_node+1):
        e_list.append((start_node + i - 1, start_node + i, dict(weight = 1)))
        # print('edge_connecting', e_list)
    return e_list


def nodes_pos_dumbbell(b, d=2, num_middle_node=0):
    if num_middle_node==-1:
        return  {**nodes_pos_polygon(b, d=d), **nodes_pos_polygon(b, d=d, offset=[2*d,0],
                                start_node=b-1, last_radian=np.pi-2*np.pi/b)}
    elif num_middle_node==0:
        return  {**nodes_pos_polygon(b, d=d), **nodes_pos_polygon(b, d=d, offset=[3*d,0],
                                start_node=b, last_radian=np.pi-2*np.pi/b)}
    else:  
        return {**nodes_pos_polygon(b, d=d), **nodes_pos_connecting(b, d=d, num_middle_node=num_middle_node),
            **nodes_pos_polygon(b, d=d, offset=[d*(num_middle_node+3),0],
                                start_node=b+num_middle_node, last_radian=np.pi-2*np.pi/b)}




def edges_dumbbell(b, num_middle_node=0, dumbbell_type='normal'):
    if dumbbell_type=='normal' or dumbbell_type=='broken':
        return edges_clique(b) + edges_connecting(b, num_middle_node=num_middle_node) + edges_clique(
            b, start_node=b+num_middle_node)
    elif dumbbell_type=="biased":
        return edges_clique(b) + edges_connecting(b, num_middle_node=num_middle_node) + edges_polygon(
            b, start_node=b+num_middle_node)


def create_dumbbell(b=b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type):
    if dumbbell_type=='normal' or dumbbell_type=='broken' or dumbbell_type=='biased':
        e_list = edges_dumbbell(b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type)    

        
    # e_list.append((b-1, b, {'weight': 1}))
    n_pos_dict = nodes_pos_dumbbell(b, num_middle_node=num_middle_node)
    G = nx.Graph()
    G.add_edges_from(e_list)
    # print(e_list, G.nodes(), print(n_pos_dict))
    for i, j in n_pos_dict.items():
        G.nodes[i]['pos'] = j
    if dumbbell_type=='broken':
        G.remove_edge(b+num_middle_node+1, 2*b+num_middle_node-1)
    return G


def plot_dumbbell(b=b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type):
    g=create_dumbbell(b,num_middle_node=num_middle_node, dumbbell_type=dumbbell_type)
    # print(g.nodes(data=True))
    # print(g.edges(data=True))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    nx.draw_networkx_nodes(g, pos=g.nodes(data='pos'), node_color='tab:blue', node_size=500)
    nx.draw_networkx_labels(g, pos=g.nodes(data='pos'), font_size=14, font_color='whitesmoke')
    labels = nx.get_edge_attributes(g,'weight')
    nx.draw_networkx_edges(g, pos=g.nodes(data='pos'), edge_color='gray', width=2)
    # nx.draw_networkx_edge_labels(g,pos=pos,edge_labels=labels)

    ax.set_aspect(1)
    plt.show()
    return g



# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets

# import matplotlib.pyplot as plt
# %matplotlib inline

# import numpy as np
widget_plot_dumbbell=interactive(plot_dumbbell, b = widgets.IntSlider(value=4, min=3, max=15, step=1),
         num_middle_node=widgets.IntSlider(value=0, min=-1, max=5, step=1, description='number of node in-between'), 
         dumbbell_type=widgets.Dropdown(options=[('normal','normal'),('biased', 'biased'),('broken', 'broken')],
                                                value='normal',
                                                description='dumbell type'))
#display(widget_plot_dumbbell)



# def initialize_variables(g, cd_min):
def initialize_variables(g, cd_min):  # keep initialize_variables to be used separately
    # g can be obtained by g=create_dumbbell(b=b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type)


    cd_solutions = {}  # cd means cardinality
    cd_criticals_sp = {}
    
    def total_internal_edges(g, nbunch):
        """nbunch: frosenset, set or list"""
        return sum([w for _,_,w in list(g.subgraph(nbunch).edges.data('weight'))])
    
    if cd_min >= 0:
        cd_min_set = set(itertools.combinations(g.nodes(), cd_min))
        community_1_intercept = max([total_internal_edges(g, c) for c in cd_min_set])
        community_1_set = {frozenset(c) for c in cd_min_set if total_internal_edges(g, c)==community_1_intercept}
        cd_solutions[cd_min] = cd_solutions.get(cd_min, set()).union(community_1_set)

    cd_solutions[len(g.nodes())] = {frozenset(g.nodes)}
    cd_criticals_sp[len(g.nodes())] = 0

    
    
    ## for each cardinality, find the best sets
    def find_candidate_for_cd(g):
        cd_intercept_dict = {}
        cd_candidate_community_dict = {}
        for cd in range(cd_min, len(g.nodes())+1):
            temp_intercept = -np.inf
            for c in set(itertools.combinations(g.nodes(), cd)):
                c_intercept =  total_internal_edges(g, c)
                if c_intercept > temp_intercept:
                    cd_candidate_community_dict.setdefault(cd, set()).clear()
                    temp_intercept = c_intercept
                    cd_intercept_dict[cd] = c_intercept
                    cd_candidate_community_dict.setdefault(cd, set()).add(frozenset(c))
                if c_intercept == temp_intercept:
                    cd_candidate_community_dict.setdefault(cd, set()).add(frozenset(c))
        return cd_intercept_dict, cd_candidate_community_dict
    cd_intercept_dict, cd_candidate_community_dict= find_candidate_for_cd(g)    


    f_sp = {}     ## the functions for the candidates sets of each cardinality
    # cd_y_intercept_sp = {}
    for cd in range(cd_min, len(g.nodes())+1):
        f_sp.setdefault(cd, - cd * x + cd_intercept_dict[cd])
    # f_sp

    info_plot = {} ## store the information for plotting
    # new_run = True

    community_1 = list(cd_solutions[cd_min])[0]
    community_2 = list(cd_solutions[len(g.nodes())])[0]    
    
    return cd_solutions, cd_criticals_sp, f_sp, info_plot, cd_intercept_dict, cd_candidate_community_dict, community_1, community_2

# cd_solutions, cd_criticals_sp, f_sp, info_plot, cd_intercept_dict, cd_candidate_community_dict, community_1, community_2 = initialize_variables(g)



# def find_critical(g, community_1=community_1, community_2=community_2, 
#                   cd_solutions = cd_solutions, cd_criticals_sp=cd_criticals_sp, new_run_reset = True):
def find_critical(g=None, cd_min=cd_min):
    # be careful when passing a list or a set to arguments
    # global new_run, info_plot
    # if new_run == True:
    #     info_plot = {}

    if g==None:
        raise Exception('g should be provided and can be obtained by g=create_dumbbell(b=b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type)')

    
    cd_solutions, cd_criticals_sp, f_sp, info_plot, cd_intercept_dict, cd_candidate_community_dict, community_1, community_2 = initialize_variables(g, cd_min)
    
    # def _find_critical(g, community_1=None, community_2=None, cd_solutions=None, cd_criticals_sp=None, 
    #                    f_sp=None, info_plot=None,cd_intercept_dict=None, cd_candidate_community_dict=None):
    def _find_critical(g, community_1=None, community_2=None):
        # community_1 = list(cd_solutions[cd_min])[0]
        # community_2 = list(cd_solutions[len(g.nodes())])[0]

        # global info_plot
        # print(info_plot)

        current_call_index = len(info_plot)
        info_plot[current_call_index] = {0:[community_1,community_2]}

        cd1 = len(community_1)
        cd2 = len(community_2)

        # f_sp[cd1] = - cd1 * x + total_internal_edges(g, community_1)
        # f_sp[cd2] = - cd2 * x + total_internal_edges(g, community_2)

        cross_sp = sp.linsolve([f_sp[cd1] - y, f_sp[cd2] - y], (x, y))
        cross_x_sp, cross_y_sp = list(list(cross_sp)[0])
        # print(cd1, cd2, cross_x_sp)

        info_plot[current_call_index][1] = [cross_x_sp, cross_y_sp]

        temp_best_cd = cd1
        temp_non_minimal_cd_set = set()
        temp_best_y_sp = cross_y_sp
        for cd in range(cd2-1, cd1-1, -1):  # from large cardinality to small cardinality
            cd_y_sp = f_sp[cd].subs(x, cross_x_sp)
            # cd1 should be computed, otherwise will cause additional non-usefull recursion when multiple line intersect together. then should be careful not to introduce solutions for cd1 into non minimal solution set
            if cd_y_sp == temp_best_y_sp: # 
                temp_best_y_sp = cd_y_sp
                temp_best_cd = cd
                if cd != cd1:
                    temp_non_minimal_cd_set.add(cd)

            if cd_y_sp > temp_best_y_sp:
                temp_best_y_sp = cd_y_sp
                temp_best_cd = cd
                temp_non_minimal_cd_set.clear()
            # print(cd, cd_y_sp, temp_best_y_sp)

        info_plot[current_call_index][2] = [False, False] # first len(temp_non_minimal_cd_set)==0, second for temp_best==cd1 
        info_plot[current_call_index][3] = [set(), set()]

        if len(temp_non_minimal_cd_set)!=0:
            for temp_cd in temp_non_minimal_cd_set:
                cd_solutions[temp_cd]=cd_solutions.setdefault(temp_cd, set()).union(cd_candidate_community_dict[temp_cd])

            info_plot[current_call_index][2][0] = True
            info_plot[current_call_index][3][0] = temp_non_minimal_cd_set

        if temp_best_cd == cd1:
            cd_solutions[cd1]=cd_solutions.setdefault(cd1, set()).union(cd_candidate_community_dict[cd1])
            cd_criticals_sp[cd1] = cross_x_sp
            # print(cross_x_sp,cd_criticals_sp)

            info_plot[current_call_index][2][1] = True        
            return
        else:
            cd_solutions[temp_best_cd]=cd_solutions.setdefault(temp_best_cd, set()).union(cd_candidate_community_dict[temp_best_cd])

            info_plot[current_call_index][2][1] = False
            info_plot[current_call_index][3][1] = temp_best_cd           

            # print(temp_best_cd, cd_solutions)
            community = list(cd_solutions[temp_best_cd])[0]
            # print('temp_best_cd', temp_best_cd, community)
            _find_critical(g, community_1, community)
            _find_critical(g, community, community_2)    
    
    
    # _find_critical(g, community_1=community_1, community_2=community_2, 
    #               cd_solutions = cd_solutions, cd_criticals_sp=cd_criticals_sp, f_sp=f_sp, info_plot=info_plot, cd_intercept_dict=cd_intercept_dict, cd_candidate_community_dict=cd_candidate_community_dict)

    # if new_run_reset == True:
    #     new_run = True
    # return cd_solutions, cd_criticals_sp, info_plot    
    
    _find_critical(g, community_1=community_1, community_2=community_2)
    return cd_solutions, cd_criticals_sp, f_sp, info_plot, cd_intercept_dict, cd_candidate_community_dict


g = create_dumbbell(b=b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type)
cd_solutions, cd_criticals_sp, f_sp, info_plot, cd_intercept_dict, cd_candidate_community_dict = find_critical(g, cd_min)

## Solve the autowrapping problem in jupyter notebook by using ax.add_artist()
## https://stackoverflow.com/a/56552098
# import matplotlib.pyplot as plt
# import matplotlib.text as mtext

class WrapText(mtext.Text):
    def __init__(self,
                 x=0, y=0, text='',
                 width=0,
                 **kwargs):
        mtext.Text.__init__(self,
                 x=x, y=y, text=text,
                 wrap=True,
                 **kwargs)
        self.width = width  # in screen pixels. You could do scaling first

    def _get_wrap_line_width(self):
        return self.width

# fig = plt.figure(1, clear=True)
# ax = fig.add_subplot(111)

# text = ('Lorem ipsum dolor sit amet, consectetur adipiscing elit, '
#         'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. ')

# # Create artist object. Note clip_on is True by default
# # The axes doesn't have this method, so the object is created separately
# # and added afterwards.
# wtxt = WrapText(.8, .4, text, width=200, va='top', clip_on=False,
#                 bbox=dict(boxstyle='square', fc='w', ec='b'))
# # Add artist to the axes
# ax.add_artist(wtxt)

# plt.show()


# %matplotlib inline
# from IPython.display import display, clear_output, Latex

# import numpy as np
# import matplotlib.pyplot as plt 
# import matplotlib as mpl
# import time
class CusPlotSetting:    
    cmap_basic = mpl.colors.ListedColormap(['orangered','blue', 'purple', 'green','red','magenta', 'blueviolet','yellowgreen', 'deeppink', 'cyan'])
    cmap = mpl.colors.ListedColormap(cmap_basic.colors * max(100, (len(g.nodes())//len(cmap_basic.colors)+1)))  ## this is based on b, better not to use
    # cmap = plt.get_cmap('Paired')

    # def _plot_step(ax, x, y, **kwargs):
    #     ax.plot(x, y, **kwargs)
    #     # plt.show()
    #     return ax

    figsize_x=16
    figsize_y=12
    font_size = 12

    fig_aspect=0.5
    fig_xlim_left=0
    fig_xlim_right=8
    fig_ylim_bottom=-18
    fig_ylim_top=13

    text_x_pos = 3.5
    linewidth_1 = 0.3
    linewidth_2 = 3
    scatter_size_1 = 10
    scatter_size_2 = 80

    zorder_1 = len(info_plot)
    zorder_2 = 10*len(info_plot)

    d_offset = abs(f_sp[cd_min].evalf(subs={x:max(cd_criticals_sp.values())})) #offset for text position based on the distance from first critical point to x axis

    xx_right=8
# xx=np.linspace(0,xx_right)

def plot_steps(step_pause = len(info_plot)*4,
               g=g, cd_solutions=cd_solutions, cd_criticals_sp=cd_criticals_sp, f_sp=f_sp, info_plot=info_plot, cd_intercept_dict=cd_intercept_dict, cd_candidate_community_dict=cd_candidate_community_dict,
                   text_x_pos=CusPlotSetting.text_x_pos, linewidth_1 = CusPlotSetting.linewidth_1, 
                   linewidth_2 = CusPlotSetting.linewidth_2, scatter_size_1 = CusPlotSetting.scatter_size_1, scatter_size_2 = CusPlotSetting.scatter_size_2, 
                   zorder_1=CusPlotSetting.zorder_1,zorder_2=CusPlotSetting.zorder_2, d_offset=CusPlotSetting.d_offset,
                   figsize_x=CusPlotSetting.figsize_x, figsize_y=CusPlotSetting.figsize_y, font_size=CusPlotSetting.font_size, fig_aspect=CusPlotSetting.fig_aspect,
                   fig_xlim_left=CusPlotSetting.fig_xlim_left, fig_xlim_right=CusPlotSetting.fig_xlim_right, 
                   fig_ylim_bottom=CusPlotSetting.fig_ylim_bottom, fig_ylim_top=CusPlotSetting.fig_ylim_top, 
                   xx_right=CusPlotSetting.xx_right, cmap_basic=CusPlotSetting.cmap_basic):
    ## ipython interact widget can't accept function with arguments with type tuple.
    
    cmap = mpl.colors.ListedColormap(cmap_basic.colors * (len(g.nodes())//len(cmap_basic.colors)+2))
    
    xx=np.linspace(0, xx_right)
    fig = plt.figure(figsize=(figsize_x, figsize_y))

    ax = fig.add_subplot(121)
    plt.rc('font', size=font_size) 

    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data',0))

    ax.set_aspect(fig_aspect)
    ax.set_xlim(fig_xlim_left, fig_xlim_right)
    ax.set_ylim(fig_ylim_bottom, fig_ylim_top)
    ax.xaxis.get_major_ticks()[0].label1.set_visible(False)  # https://stackoverflow.com/a/13583251
    
    
    ax2 = fig.add_subplot(122)
    pos=g.nodes('pos')
    nx.draw_networkx_nodes(g, pos=g.nodes(data='pos'), node_color='tab:blue', node_size=500, ax=ax2)
    nx.draw_networkx_labels(g, pos=g.nodes(data='pos'), font_size=14, font_color='whitesmoke',ax=ax2)
    nx.draw_networkx_edges(g, pos=g.nodes(data='pos'), edge_color='gray', width=2, ax=ax2)
    ax2.set_aspect(1)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    print('If there are multiple solutions, a random solution is indicated on the right hand side.')
    
    
    step_current = 0

    for big_step in range(0, len(info_plot)):
        info = info_plot[big_step]
        cd1, cd2 = len(info[0][0]), len(info[0][1])
        yy1 = - cd1 * xx + cd_intercept_dict[cd1]
        yy2 = - cd2 * xx + cd_intercept_dict[cd2]

        if step_pause==0: break ## 

        
        f2_str = 'y='+str(f_sp[cd2])
        ax.plot(xx, yy2, label=f2_str, color=cmap.colors[cd2], linewidth=linewidth_1)
        ax.text(text_x_pos, f_sp[cd2].evalf(subs={x:text_x_pos}) +0.1, f2_str, color=cmap.colors[cd2])

        f1_str = 'y='+str(f_sp[cd1])
        ax.plot(xx, yy1, label='y='+str(f_sp[cd1]), color=cmap.colors[cd1], linewidth=linewidth_1)
        ax.text(text_x_pos, f_sp[cd1].evalf(subs={x:text_x_pos}) +0.1, f1_str, color=cmap.colors[cd1])
        # kwargs = {'label':'y='+str(f_sp[cd1])}
        # ax = _plot_step(ax, xx, yy1, **kwargs)
        # plt.show()
        # time.sleep(2)
        
        step_current += 1
        if step_current >= step_pause: break

        ax.scatter(*info[1], s=scatter_size_1, c=cmap.colors[cd1])
        
        step_current += 1
        if step_current >= step_pause: break

        
        if info[2][0] == True:
            for cdprime in info[3][0]:
                yyprime = - cdprime * xx + cd_intercept_dict[cdprime]
                ax.plot(xx, yyprime, label='y='+str(f_sp[cdprime]), color=cmap.colors[cdprime], linewidth=linewidth_1)
        
            step_current += 1
            if step_current >= step_pause: break


        if info[2][1] == True:
            ax.scatter(*info[1], s=scatter_size_2, c=cmap.colors[cd1], zorder=zorder_2)
            

        
            step_current += 1
            if step_current >= step_pause: break


            
            # if d_offset==None: This is restricted to recursion from small community to large community
            #     d_offset = (0 if info[1][1]>0 else - info[1][1])
            #     # print(d_offset)
            point_text = [info[1][0],  info[1][1] + d_offset + 3]
            ax.plot([info[1][0], point_text[0]], [info[1][1], point_text[1]], color=cmap.colors[cd1], ls='dashed', linewidth=linewidth_2)

            cd1_solution_text = ', '.join(str(set(s)) for s in cd_solutions[cd1]) if cd1!=0 else str(r'$\emptyset$')
            text = str('x=') + str(info[1][0]) + str(': ') + str(cd1_solution_text)
            wtxt = WrapText(point_text[0]+0.3,point_text[1], text, color=cmap.colors[cd1], width=200, va='top', clip_on=True,
                    bbox=dict(boxstyle='square', fc='w', ec=cmap.colors[cd1], lw=0))
            ax.add_artist(wtxt)

            if big_step > 0:
                if cd1 != len(info_plot[big_step-1][0][0]):
                    cross_x_cd1_with_last_cd1 = float(list(list(
                        sp.linsolve([f_sp[cd1] - y, f_sp[len(info_plot[big_step-1][0][0])] - y], (x,y)))[0])[0])
                else:
                    cross_x_cd1_with_last_cd1 = np.inf
            if big_step == 0:
                cross_x_cd1_with_last_cd1 = np.inf
            # here can be better, want to decide the boundary for current cd1 curve to be a solution, and 
            # currently in the recursion, we always calculate from smaller community_1, hence can take it from 
            # the interception of community_1 with that in last call of the recursion.
            # if the community_1 in the last call is the most smallest cardinality, current community_1 will be the same, so can't get an intersection soution

            xx_cd1=np.linspace(float(info[1][0]), min(max(xx), cross_x_cd1_with_last_cd1))
            yy_cd1=- cd1 * xx_cd1 + cd_intercept_dict[cd1]
            ax.plot(xx_cd1, yy_cd1, color=cmap.colors[cd1], linewidth=linewidth_2)
            
            
            nx.draw_networkx_nodes(g, pos=g.nodes(data='pos'), node_color='tab:blue', node_size=500, ax=ax2)
            nx.draw_networkx_nodes(list(cd_solutions[cd1])[0], pos=g.nodes(data='pos'), node_color=cmap.colors[cd1], node_size=500, ax=ax2)
            
            nx.draw_networkx_labels(g, pos=g.nodes(data='pos'), font_size=14, font_color='whitesmoke',ax=ax2)
            nx.draw_networkx_edges(g, pos=g.nodes(data='pos'), edge_color='gray', width=2, ax=ax2)
            # ax2.add_text(0.5*ax2.bbox.width, 0.2*ax2.bbox.height, str('If there are multiple solutions, a random solution is indicated on the right hand side.'), color=cmap.colors[cd1])

            
            
            

            if cd2 == len(info_plot[0][0][1]): # the end must be the one related with the line for V
                
                step_current += 1
                if step_current >= step_pause: break
                
                xx_cd2=np.linspace(0, float(info[1][0]))
                yy_cd2=- cd2 * xx_cd2 + cd_intercept_dict[cd2]
                ax.plot(xx_cd2, yy_cd2, color=cmap.colors[cd2], linewidth=linewidth_2)
                ax.text(0+0.5, cd_intercept_dict[cd2] -0.25*cd2, str(set(list(cd_solutions[cd2])[0])), color=cmap.colors[cd2])              
                
                nx.draw_networkx_nodes(g, pos=g.nodes(data='pos'), node_color='tab:blue', node_size=500, ax=ax2)
                nx.draw_networkx_nodes(list(cd_solutions[cd2])[0], pos=g.nodes(data='pos'), node_color=cmap.colors[cd2], node_size=500, ax=ax2)
                nx.draw_networkx_labels(g, pos=g.nodes(data='pos'), font_size=14, font_color='whitesmoke',ax=ax2)
                nx.draw_networkx_edges(g, pos=g.nodes(data='pos'), edge_color='gray', width=2, ax=ax2)


    # ax2 = fig.add_subplot(122)
    # ax2.axis([0,10,0,10])
    # ax2_str = str(cd_criticals_sp_list) + str(cd_solutions)
    # ## https://stackoverflow.com/a/15740730
    # from textwrap import wrap
    # # ax2_str_wrapped = [ '\n'.join(wrap(l, 20)) for l in ax2_str ]
    # # ax2_text = ax2.text(2, 10, ax2_str, color=cmap.colors[cd2], ha='left', wrap=True )    
    # # ## https://stackoverflow.com/a/56552098
    # # ax2_text._get_wrap_line_width = lambda : 600.
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['bottom'].set_visible(False)
    # ax2.spines['left'].set_visible(False)
    # ax2.set_xticks([])
    # ax2.set_yticks([])

    # for i, slope in enumerate(slope_list):
    #     slope_in_label = '' if slope_list[i] == 1 else str(slope_list[i])
    #     if intercept_list[i]>0:
    #         intercept_in_label = ' + ' + str(intercept_list[i])
    #     elif intercept_list[i]<0:
    #         intercept_in_label = ' - ' + str(-intercept_list[i])
    #     else:
    #         intercept_in_label = ''
    #     locals()['y_'+str(slope)] = slope_in_label + 'x' + intercept_in_label
    #     y=slope * x + intercept_list[i]
    #     plt.plot(x, y, label='y=' + locals()['y_'+str(slope)])
    # plt.legend()
    plt.show()


def total_steps_by_info_plot(info_plot=None): # should according to plot_steps()
    if info_plot==None:
        raise Exception('The variable info_plot is None!')
    steps_total=0
    for big_step in range(0, len(info_plot)):
        steps_total += info_plot[big_step][2][0] + info_plot[big_step][2][1]
    steps_total += 2*len(info_plot)+1+1
    return steps_total
# but interact widget do not when info_plot changed due to the change of b, this value can't be updated automaticly. Hence should need another widget to help, or create a global variable or store in a class.


def plot_steps_with_important_parameter(step_pause, text_x_pos=CusPlotSetting.text_x_pos,
                  figsize_x=CusPlotSetting.figsize_x, figsize_y=CusPlotSetting.figsize_y, font_size=CusPlotSetting.font_size, fig_aspect=CusPlotSetting.fig_aspect, 
                                        fig_xlim_right=CusPlotSetting.fig_xlim_right, fig_ylim_bottom=CusPlotSetting.fig_ylim_bottom, fig_ylim_top=CusPlotSetting.fig_ylim_top, xx_right=CusPlotSetting.xx_right):
    plot_steps(step_pause, text_x_pos=text_x_pos,
                  figsize_x=figsize_x, figsize_y=figsize_y, font_size=font_size, fig_aspect=fig_aspect, 
                                        fig_xlim_right=fig_xlim_right, fig_ylim_bottom=fig_ylim_bottom, fig_ylim_top=fig_ylim_top, xx_right=xx_right)
    return 

widget_plot_steps_with_important_parameter = interactive(plot_steps_with_important_parameter, step_pause=widgets.IntSlider(value=3, min=0, max=len(info_plot)*4, step=1, description='Calculation steps'))
controls = HBox(widget_plot_steps_with_important_parameter.children[:-1], layout = Layout(flex_flow='row wrap'))
output = widget_plot_steps_with_important_parameter.children[-1]
#display(VBox([controls, output]))




def plot_steps_with_b(b=b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type, cd_min=cd_min, 
                      step_pause=3, text_x_pos=CusPlotSetting.text_x_pos,
                  figsize_x=CusPlotSetting.figsize_x, figsize_y=CusPlotSetting.figsize_y, font_size=CusPlotSetting.font_size, fig_aspect=CusPlotSetting.fig_aspect, 
                                        fig_xlim_right=CusPlotSetting.fig_xlim_right, fig_ylim_bottom=CusPlotSetting.fig_ylim_bottom, fig_ylim_top=CusPlotSetting.fig_ylim_top, xx_right=CusPlotSetting.xx_right):
    g=create_dumbbell(b=b, num_middle_node=num_middle_node, dumbbell_type=dumbbell_type)
    cd_solutions, cd_criticals_sp, f_sp, info_plot, cd_intercept_dict, cd_candidate_community_dict = find_critical(g, cd_min)

    plot_steps(step_pause=step_pause,  g=g, cd_solutions=cd_solutions, cd_criticals_sp=cd_criticals_sp, f_sp=f_sp, info_plot=info_plot, cd_intercept_dict=cd_intercept_dict, cd_candidate_community_dict=cd_candidate_community_dict,
                   text_x_pos=text_x_pos,
                  figsize_x=figsize_x, figsize_y=figsize_y, font_size=font_size, fig_aspect=fig_aspect, 
                                        fig_xlim_right=fig_xlim_right, fig_ylim_bottom=fig_ylim_bottom, fig_ylim_top=fig_ylim_top, xx_right=xx_right)
    return 



widget_plot_steps_with_b = interactive(plot_steps_with_b, b=widgets.IntSlider(value=4, min=3, max=8, step=1, description='b'),
                                       num_middle_node=widgets.IntSlider(value=0, min=-1, max=6, step=1, description='Node in-between'), 
                                       dumbbell_type=widgets.Dropdown(options=[('normal','normal'),('biased', 'biased'),('broken', 'broken')],
                                                value='normal',
                                                description='dumbell type'),
                                       cd_min=widgets.IntSlider(value=1, min=0, max=5, step=1, description='k'),
                                       step_pause=widgets.IntSlider(value=3, min=0, max=10*b, step=1, description='Steps'))
controls = HBox(widget_plot_steps_with_b.children[:-1], layout = Layout(flex_flow='row wrap'))
output = widget_plot_steps_with_b.children[-1]
#display(VBox([controls, output]))



######

def playground(b=3, k=1, m=0, t="normal"):
    widget_plot_steps_with_b = interactive(
        plot_steps_with_b,
        b=widgets.IntSlider(value=b, min=3, max=8, step=1, description="b"),
        num_middle_node=widgets.IntSlider(
            value=m, min=-1, max=6, step=1, description="Node in-between"
        ),
        dumbbell_type=widgets.Dropdown(
            options=[("normal", "normal"), ("biased", "biased"), ("broken", "broken")],
            value=t,
            description="dumbell type",
        ),
        cd_min=widgets.IntSlider(value=k, min=0, max=5, step=1, description="k"),
        step_pause=widgets.IntSlider(
            value=5 * b, min=0, max=5 * b, step=1, description="steps"
        ),
    )
    controls = HBox(
        widget_plot_steps_with_b.children[:-1], layout=Layout(flex_flow="row wrap")
    )
    output = widget_plot_steps_with_b.children[-1]
    display(VBox([controls, output]))


def dumbbell(b=3, m=0, t="normal"):
    widget_plot_dumbbell = interactive(
        plot_dumbbell,
        b=widgets.IntSlider(value=b, min=3, max=15, step=1),
        num_middle_node=widgets.IntSlider(
            value=m, min=-1, max=5, step=1, description="number of node in-between"
        ),
        dumbbell_type=widgets.Dropdown(
            options=[("normal", "normal"), ("biased", "biased"), ("broken", "broken")],
            value=t,
            description="dumbell type",
        ),
    )
    display(widget_plot_dumbbell)
    
    
    
#### 
def playground_2(b=3, k=1, m=0, t="normal"):
    widget_plot_steps_with_b = interactive(plot_steps_with_b, 
                                           b=widgets.IntSlider(value=b, min=3, max=8, step=1, description='b'),
                                           num_middle_node=widgets.IntSlider(value=m, min=-1, max=6, step=1, description="number of node in-between", style={'description_width': 'initial'}), 
                                           dumbbell_type=widgets.Dropdown(options=[('normal','normal'),('biased', 'biased'),('broken', 'broken')], value=t, description='dumbell type'),
                                           cd_min=widgets.IntSlider(value=k, min=0, max=5, step=1, description='k'),
                                           step_pause=widgets.IntSlider(value=5*b, min=0, max=5*b, step=1, description='Steps', layout=Layout(width='50%')))
    # output_observe = widgets.Output()
    def update_widget_plot_steps_with_b(*args):
        # with output_observe:
        #     print('observe take effect')
        _, _, _, info_plot, _, _ = find_critical(create_dumbbell(b=widget_plot_steps_with_b.children[0].value, num_middle_node=widget_plot_steps_with_b.children[1].value, 
                                                                 dumbbell_type=widget_plot_steps_with_b.children[2].value), widget_plot_steps_with_b.children[3].value)
        # _, _, _, info_plot, _, _ = find_critical(create_dumbbell(b=widget_b.value, num_middle_node=widget_plot_steps_with_b.children[1].value, dumbbell_type=widget_dumbbell_type.value), widget_cd_min.value)
        # with output_observe:
        #     print(info_plot)
        #     print(total_steps_by_info_plot(info_plot=info_plot))
        widget_plot_steps_with_b.children[4].max = total_steps_by_info_plot(info_plot=info_plot)    


    # widget_plot_steps_with_b.children[1].observe(update_widget_plot_steps_with_b, 'value')

    for child in widget_plot_steps_with_b.children[0:4]:
        child.observe(update_widget_plot_steps_with_b, 'value')


    widget_plot_steps_with_b_controls = HBox(widget_plot_steps_with_b.children[:-1], layout = Layout(flex_flow='row wrap'))
    widget_plot_steps_with_b_output = widget_plot_steps_with_b.children[-1]
    display(VBox([widget_plot_steps_with_b_controls, widget_plot_steps_with_b_output]))