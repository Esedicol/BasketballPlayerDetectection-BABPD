# from tkinter import *
# from tkinter import filedialog, Text
# from PIL import Image


# import matplotlib.pyplot as plt
# from matplotlib.pyplot import Figure
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# import tensorflow as tf

# # from tfp import ball_tracking_main

# from YOLO import yolo

# def openDialog():
# 	root.filename = filedialog.askopenfilename(
# 		initialdir="/Users/esedicol/Desktop/Desktop/Basketball-Shot-Detectection", 
# 		title="Select a Video File", 
# 		filetypes=(("mov files", "*.mov"), ("mp4 files", "*.mp4")))
# 	file_name = root.filename
# 	print(file_name)

# from matplotlib.patches import Circle, Rectangle, Arc

# def draw_court(ax=None, color='black', lw=2, outer_lines=False):
#     # If an axes object isn't provided to plot onto, just get current one
#     if ax is None:
#         ax = plt.gca()

#     # Create the various parts of an NBA basketball court

#     # Create the basketball hoop
#     # Diameter of a hoop is 18" so it has a radius of 9", which is a value
#     # 7.5 in our coordinate system
#     hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

#     # Create backboard
#     backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

#     # The paint
#     # Create the outer box 0f the paint, width=16ft, height=19ft
#     outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
#                           fill=False)
#     # Create the inner box of the paint, widt=12ft, height=19ft
#     inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
#                           fill=False)

#     # Create free throw top arc
#     top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
#                          linewidth=lw, color=color, fill=False)
#     # Create free throw bottom arc
#     bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
#                             linewidth=lw, color=color, linestyle='dashed')
#     # Restricted Zone, it is an arc with 4ft radius from center of the hoop
#     restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
#                      color=color)

#     # Three point line
#     # Create the side 3pt lines, they are 14ft long before they begin to arc
#     corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
#                                color=color)
#     corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
#     # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
#     # I just played around with the theta values until they lined up with the 
#     # threes
#     three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
#                     color=color)

#     # Center Court
#     center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
#                            linewidth=lw, color=color)
#     center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
#                            linewidth=lw, color=color)

#     # List of the court elements to be plotted onto the axes
#     court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
#                       bottom_free_throw, restricted, corner_three_a,
#                       corner_three_b, three_arc, center_outer_arc,
#                       center_inner_arc]

#     if outer_lines:
#         # Draw the half court line, baseline and side out bound lines
#         outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
#                                 color=color, fill=False)
#         court_elements.append(outer_lines)

#     # Add the court elements onto the axes
#     for element in court_elements:
#         ax.add_patch(element)

#     return ax

# def plot_positions():
# 	plt.figure(figsize=(8,7))
# 	draw_court(outer_lines=True)
# 	plt.xlim(-300,300)
# 	plt.ylim(-100,500)
# 	plt.axis('off')
# 	plt.show()

# file_name = ""

# root = Tk()
# root.configure(background='#00261C',padx=10, pady=10)
# root.geometry("1020x570")
# root.title("Basketball Shot Detection")

# # --------------------- Frames --------------------- #


# # --------------------- LEFT COLUMN --------------------- #
# left_frame = Frame(root, width=500, height=500, background="Blue")
# left_frame.grid(row=0, column=0)

# # --------------------- RIGHT COLUMN --------------------- #
# right_frame = Frame(root, width=500, height=500, background="Green")
# right_frame.grid(row=0, column=1)


# # --------------------- INNER RIGHT COLUMN --------------------- #
# top_rFrame = Frame(right_frame, width=500, height=400, background="Red")
# top_rFrame.grid(row=0, column=0)

# bottom_rFrame = Frame(right_frame, width=500, height=100, background="Yellow")
# bottom_rFrame.grid(row=1, column=0)


# # --------------------- BOTTOM COLUMN --------------------- #
# lButtom_frame = Frame(root, width=500, height=50, background="#00261C")
# lButtom_frame.grid(row=1, column=0)

# rButtom_frame = Frame(root, width=500, height=50, background="#00261C")
# rButtom_frame.grid(row=1, column=1)

# open_file_button = Button(lButtom_frame, text="Open File", command=openDialog,padx=200, fg="Black")
# open_file_button.place(relx=0.5, rely=0.5, anchor=CENTER)

# # photo = PhotoImage(file="open_button.png")
# delete_file_button = Button(rButtom_frame, text="Open File", command=plot_positions,padx=200)
# delete_file_button.place(relx=0.5, rely=0.5, anchor=CENTER)

# root.mainloop()


import tkinter

main = tkinter.Tk()
main.geometry("800x600")
main.configure(background='#00261C',padx=10, pady=10)
main.title("Momentum")

main.grid_rowconfigure(5, weight=1)
main.grid_columnconfigure(0, weight=1)
main.grid_columnconfigure(1, weight=1)

canvas = tkinter.Canvas(main, width=700, height=300)
ol1= tkinter.Label(main, text="Object A")
ol2= tkinter.Label(main,text="Object B")
ml1 = tkinter.Label(main, text ="MASS")
ml2 = tkinter.Label(main, text ="MASS")
me1 = tkinter.Entry(main)
me2 = tkinter.Entry(main)

canvas.grid(row=0,column=0, columnspan=2)
ol1.grid(row=1, column=0, pady=(0,20))
ol2.grid(row=1, column=1, pady=(0,20))
ml1.grid(row=2, column=0)
ml2.grid(row=2, column=1)
me1.grid(row=3, column=0)
me2.grid(row=3, column=1)


main.mainloop()