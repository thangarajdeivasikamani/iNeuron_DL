import matplotlib.pyplot as plt

def plot_single_digit_element(shape_of_digit):
   
    plt.imshow(shape_of_digit,cmap='binary')
    plt.axis("off")
    plt.savefig(".\\Plot\\number.png")
    print(f"image min size:{shape_of_digit.min()}")
    print(f"image max size:{shape_of_digit.max()}")

def accuracy_plt(hist_dataframe):
   hist_dataframe.plot(figsize=(10,7))
   plt.grid(True)
   plt.savefig(".\\Plot\\accuracy_plt.png")

def plot_predicted(X_pred, Y_pred, y_test):
    count = 1
    for img_array, pred, actual in zip(X_pred, Y_pred, y_test):
        plt.imshow(img_array, cmap="binary")
        plt.title(f"predicted: {pred}, Actual: {actual}")
        plt.axis("off")
        plt.savefig(".\\Plot\\predicted_number"+str(count)+".png")
        count = count + 1