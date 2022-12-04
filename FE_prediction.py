print("\nLoading Training data . . . ")

X_train, y_train, X_valid, y_valid = load_and_preprocess_data(**pipeline["training_data"])

print("Loaded Training data Sucessfully")
print("the shape of X_train: ", X_train.shape)
print("the shape of y_train: ", y_train.shape)
print("the shape of X_valid: ", X_valid.shape)
print("the shape of y_valid: ", y_valid.shape)


print("\nLoading Testing data . . . ")

X_test, y_test = load_and_preprocess_data(**pipeline["testing_data"])

print("Loaded Testing data Sucessfully")
print("the shape of X_test: ", X_test.shape)
print("the shape of y_test: ", y_test.shape)

print("\nLoading Noisy testing data . . . ")

X_noisy_test, y_noisy_test, = load_and_preprocess_data(**pipeline["noisy_testing_data"])

print("Loaded Noisy testing data Sucessfully")
print("the shape of X_noisy_test: ", X_noisy_test.shape)
print("the shape of y_noisy_test: ", y_noisy_test.shape)