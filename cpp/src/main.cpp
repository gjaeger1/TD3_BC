#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include <filesystem>

int main(int argc, const char* argv[]) {
  // parse command line arguments with 
    // --actor
    // --critic

  if(argc != 5) {
    std::cerr << "Usage: td3_bc --actor <actor_model_path> --critic <critic_model_path>\n";
    return -1;
  }

  // loop through arguments to detect flags and set paths
  std::filesystem::path actor_path;
  std::filesystem::path critic_path;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--actor") {
      if (i + 1 < argc) { // Make sure we aren't at the end of argv!
        actor_path = argv[i + 1]; // Increment 'i' so we don't get the argument as the next argv[i].
        i++;
      } else { // Uh-oh, there was no argument to the destination option.
        std::cerr << "--actor option requires one argument." << std::endl;
        return -1;
      }  
    } else if (std::string(argv[i]) == "--critic") {
      if (i + 1 < argc) { // Make sure we aren't at the end of argv!
        critic_path = argv[i + 1]; // Increment 'i' so we don't get the argument as the next argv[i].
        i++;
      } else { // Uh-oh, there was no argument to the destination option.
        std::cerr << "--critic option requires one argument." << std::endl;
        return -1;
      }  
    }
  }

  constexpr std::size_t state_dim = 18;
  constexpr std::size_t action_dim = 2;

  torch::jit::script::Module actor;
  torch::jit::script::Module critic;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    actor = torch::jit::load(actor_path);
    critic = torch::jit::load(critic_path);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    std::cerr << "error: " << e.what() << '\n';
    return -1;
  }

  std::vector<torch::jit::IValue> critic_input;
  critic_input.push_back(torch::ones({1, state_dim}));
  
  //std::vector<torch::jit::IValue> critic_input2;
  critic_input.push_back(torch::ones({1, action_dim}));

  std::vector<torch::jit::IValue> actor_input;
  actor_input.push_back(torch::ones({1, state_dim}));

  // Execute the model and turn its output into a tensor.
  auto critic_output = critic.forward(critic_input);//.toTensor();
  // print type of critic_output
  std::cout << "Critic output type: " << critic_output.type()->str() << '\n';
  std::cout << "Critic output: " << critic_output << '\n';

  at::Tensor actor_output = actor.forward(actor_input).toTensor();
  std::cout << "Actor output: " << actor_output << '\n';

  // std::vector<torch::jit::IValue> critic_input;
  //   critic_input.push_back(torch::ones({1, state_dim+action_dim}));
    
  //   std::vector<torch::jit::IValue> actor_input;
  //   actor_input.push_back(torch::ones({1, state_dim}));

  //   // Execute the model and turn its output into a tensor.
  //   at::Tensor critic_output = critic.forward(critic_input).toTensor();
  //   std::cout << "Critic output: " << critic_output << '\n';

  //   at::Tensor actor_output = actor.forward(actor_input).toTensor();
  //   std::cout << "Actor output: " << actor_output << '\n';

  std::cout << "ok\n";
}