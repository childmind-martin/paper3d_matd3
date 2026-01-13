function paper3D()
    % 初始化参数
    clc;
    clear;
    close all;

    % 地图和参数设置
    map_size = 100;
    [X, Y] = meshgrid(1:map_size, 1:map_size);
    terrain = generateTerrain(map_size);

    % 障碍物（雷达模型）
    obstacles = [
        struct('center', [20, 30, 0], 'radius', 10); % 初始高度设为 0
        struct('center', [50, 50, 0], 'radius', 15);
        struct('center', [70, 80, 0], 'radius', 12)
    ];

    % 调整障碍物高度使其贴合地形
    for i = 1:length(obstacles)
        obstacles(i).center(3) = interp2(X, Y, terrain, obstacles(i).center(1), obstacles(i).center(2));
    end

    % 起点和终点
    start_point = [10, 10, 0];
    goal_point = [90, 90, 0];

    % 将起点和终点贴合地形
    start_point(3) = interp2(X, Y, terrain, start_point(1), start_point(2)) + 10;
    goal_point(3) = interp2(X, Y, terrain, goal_point(1), goal_point(2)) + 10;

    % RRT 参数
    max_iter = 10000;
    step_size = 3;
    redundancy_distance = 8;

    % 调用混合策略的 RRT 方法
    [path, success] = hybridRRT(start_point, goal_point, obstacles, terrain, X, Y, max_iter, step_size, redundancy_distance);

    if success
        disp('路径规划成功！');
    else
        error('路径规划失败，未找到可行路径');
    end

    % 绘制结果
    plot3DPath(path, obstacles, terrain, X, Y, start_point, goal_point);
end

%% 混合策略 RRT
function [path, success] = hybridRRT(start, goal, obstacles, terrain, X, Y, max_iter, step_size, redundancy_distance)
    % 初始化起点和终点树
    start_tree = [struct('position', start, 'parent', 0)];
    goal_tree = [struct('position', goal, 'parent', 0)];

    success = false;

    % 地形复杂度窗口大小
    window_size = 50;

    for iter = 1:max_iter
        % 获取当前位置
        current_pos = start_tree(end).position;
        x = round(current_pos(1));
        y = round(current_pos(2));

        % 计算地形复杂度
        complexity = calculateTerrainComplexity(terrain, x, y, window_size);

        strategy = selectStrategy(complexity);

        % 选择采样策略
        switch strategy
            case 1 % 人工势场引导采样
                rand_point = potentialFieldSample(current_pos, goal, obstacles, step_size);
                % 扩展起点树
                [new_start_node, success1] = extendTree(start_tree, rand_point, obstacles, step_size, terrain, X, Y);
                success2 = false; % 确保 success2 被定义
                if success1
                    start_tree = [start_tree, new_start_node];
                end
            case 2 % 随机采样
                rand_point = [randi([1, size(terrain, 1)]), randi([1, size(terrain, 2)]), randi([10, 50])];
                % 扩展起点树
                [new_start_node, success1] = extendTree(start_tree, rand_point, obstacles, step_size, terrain, X, Y);
                success2 = false; % 确保 success2 被定义
                if success1
                    start_tree = [start_tree, new_start_node];
                end
            case 3 % 双向采样
                rand_point = moveTowardsGoal(current_pos, goal, step_size);% 从当前位置向目标移动一个步长
                % 扩展起点树和终点树
                [new_start_node, success1] = extendTree(start_tree, rand_point, obstacles, step_size, terrain, X, Y);
                [new_goal_node, success2] = extendTree(goal_tree, rand_point, obstacles, step_size, terrain, X, Y);
                if success1
                    start_tree = [start_tree, new_start_node];
                end
                if success2
                    goal_tree = [goal_tree, new_goal_node];
                end
        end

        % 检查是否可以连接两棵树
        if success1 && success2 && norm(new_start_node.position - new_goal_node.position) < step_size
            % 确保路径通过新生成的节点连接
            path_start = constructPath(start_tree, numel(start_tree));
            path_goal = constructPath(goal_tree, numel(goal_tree));
            path = [path_start; flipud(path_goal)];
            success = true;
            return;
        end
    end

    path = [];
end

function strategy = selectStrategy(complexity)
    % 根据障碍物复杂程度选择策略
    if complexity < 0.2
        probabilities = [0.4, 0.2, 0.4]; % 人工势场、 随机采样、双向 RRT
    elseif complexity < 0.5
        probabilities = [0.6, 0.1, 0.3];
    else
        probabilities = [0.65, 0.2, 0.15];
    end

    cumulativeProb = cumsum(probabilities);
    r = rand(); % 生成一个0到1之间的随机数
    strategy = find(r <= cumulativeProb, 1, 'first');
end

%% 复杂度计算函数
%% 复杂度计算函数
function complexity = calculateTerrainComplexity(terrain, x, y, window_size)
    % 内置参数
    lambda_p = 1.0; % 可调系数
    Ru = 1.0; % 障碍物的影响半径
    p = 2; % 指数参数
    q = 2; % 指数参数
    r = 2; % 指数参数
    
    % 地形复杂度计算
    half_window = floor(window_size / 2);
    [rows, cols] = size(terrain);
    
    % 限制窗口范围
    x_min = max(1, x - half_window);
    x_max = min(cols, x + half_window);
    y_min = max(1, y - half_window);
    y_max = min(rows, y + half_window);
    
    % 提取局部地形
    local_terrain = terrain(y_min:y_max, x_min:x_max);
    
    % 计算每个障碍物对当前节点的影响
    obstacles = {}; % 假设障碍物信息在此处定义
    N = length(obstacles);
    influence = zeros(1, N);
    for k = 1:N
        obstacle = obstacles{k};
        x_o = obstacle(1);
        y_o = obstacle(2);
        z_o = obstacle(3);
        a_o = obstacle(4);
        b_o = obstacle(5);
        c_o = obstacle(6);
        
        S_k = ((x - x_o + Ru) / a_o)^2 * p + ((y - y_o + Ru) / b_o)^2 * q + ((z - z_o + Ru) / c_o)^2 * r;
        product = 1;
        for i = 1:N
            if i ~= k
                obstacle_i = obstacles{i};
                x_i = obstacle_i(1);
                y_i = obstacle_i(2);
                z_i = obstacle_i(3);
                a_i = obstacle_i(4);
                b_i = obstacle_i(5);
                c_i = obstacle_i(6);
                
                S_i = ((x - x_i + Ru) / a_i)^2 * p + ((y - y_i + Ru) / b_i)^2 * q + ((z - z_i + Ru) / c_i)^2 * r;
                product = product * (S_i - 1);
            end
        end
        influence(k) = product / (S_k - 1);
    end
    
    % 计算复杂度
    complexity = lambda_p * sum(influence);
end

function point = potentialFieldSample(current_pos, goal, obstacles, step_size)
    % 计算目标吸引力
    attraction = (goal - current_pos) / norm(goal - current_pos);

    % 计算障碍物排斥力
    repulsion = [0, 0, 0];
    for i = 1:length(obstacles)
        obstacle_pos = obstacles(i).center;
        distance = norm(current_pos - obstacle_pos);
        if distance < obstacles(i).radius
            repulsion = repulsion - (obstacle_pos - current_pos) / distance^2;
        end
    end

    % 合力方向
    force = attraction + repulsion;
    force = force / norm(force);

    % 生成新的采样点
    point = current_pos + force * step_size;
end

%% 其他函数保持不变
function terrain = generateTerrain(map_size)
    % 使用改进的随机山脉模型生成地形
    [X, Y] = meshgrid(1:map_size, 1:map_size);
    terrain = zeros(size(X));
    
    % 添加山脉
    num_mountains = 1; % 山脉数量
    for i = 1:num_mountains
        center_x = randi([1, map_size]); % 随机山脉中心X
        center_y = randi([1, map_size]); % 随机山脉中心Y
        height = randi([30, 70]); % 随机山脉高度（调整幅度更大）
        width = randi([25, 50]); % 随机山脉宽度（增加宽度范围）
        mountain = height * exp(-((X - center_x).^2 + (Y - center_y).^2) / (2 * width^2));
        terrain = terrain + mountain;
    end
    
    % 使用低频噪声使地形更自然
    low_freq_noise = imresize(randn(10), [map_size, map_size], 'bicubic');
    kernel = fspecial('average', [15 15]); % 创建一个 15x15 的均值滤波器
    low_freq_noise = imfilter(low_freq_noise, kernel);
    terrain = terrain + 10 * low_freq_noise;
    
    % 确保地形为正值
    terrain = max(terrain, 0);
    
    % 限制地形高度
    terrain = min(terrain, 100); % 地形高度不能超过100
end

function goal_point = moveTowardsGoal(current_position, goal_position, step_size)
    % 从当前点向目标点移动一个步长
    direction = goal_position - current_position;
    direction = direction / norm(direction); % 单位化方向
    goal_point = current_position + step_size * direction;
end

function [newNode, success] = extendTree(tree, rand_point, obstacles, step_size, terrain, X, Y)
    % 找到距离随机点最近的树节点
    distances = arrayfun(@(node) norm(node.position - rand_point), tree);
    [~, closest_idx] = min(distances);
    closest_node = tree(closest_idx);

    % 沿最近点朝随机点方向扩展
    direction = rand_point - closest_node.position;
    direction = direction / norm(direction);
    new_pos = closest_node.position + step_size * direction;

    % 检查新点是否有效
    if isValidPoint(new_pos, obstacles, terrain, X, Y)
        newNode = struct('position', new_pos, 'parent', closest_idx);
        success = true;
    else
        newNode = struct('position', [], 'parent', []);
        success = false;
    end
end

function valid = isValidPoint(pos, obstacles, terrain, X, Y)
    % 检查是否在地形范围内
    if pos(1) < 1 || pos(1) > size(terrain, 1) || ...
       pos(2) < 1 || pos(2) > size(terrain, 2) || ...
       pos(3) < 0 || pos(3) > 100
        valid = false;
        return;
    end

    % 检查是否与障碍物冲突
    for i = 1:length(obstacles)
        if norm(pos - obstacles(i).center) < obstacles(i).radius
            valid = false;
            return;
        end
    end

    % 检查是否高于地形
    terrain_height = interp2(X, Y, terrain, pos(1), pos(2));
    if pos(3) < terrain_height
        valid = false;
        return;
    end

    valid = true;
end

function path = constructPath(tree, idx)
    % 根据树结构从末尾节点回溯路径
    path = [];
    while idx > 0
        path = [tree(idx).position; path];
        idx = tree(idx).parent;
    end
end

function plot3DPath(path, obstacles, terrain, X, Y, start_point, goal_point)
    figure;
    hold on;

    % 绘制地形
    surf(X, Y, terrain, 'EdgeColor', 'none', 'FaceAlpha', 0.8);
    colormap(parula);
    colorbar;

    % 绘制障碍物（雷达模型）
    for i = 1:length(obstacles)
        [ox, oy, oz] = sphere(20);
        obstacle_x = obstacles(i).center(1);
        obstacle_y = obstacles(i).center(2);
        obstacle_z = obstacles(i).center(3);
        obstacle_radius = obstacles(i).radius;
        surf(obstacle_x + ox * obstacle_radius, ...
             obstacle_y + oy * obstacle_radius, ...
             obstacle_z + oz * obstacle_radius, ...
             'FaceColor', 'red', 'EdgeColor', 'none');
    end

    % 绘制起点和终点
    scatter3(start_point(1), start_point(2), start_point(3), 100, 'green', 'filled');
    scatter3(goal_point(1), goal_point(2), goal_point(3), 100, 'blue', 'filled');

    % 绘制路径
    if ~isempty(path)
        plot3(path(:, 1), path(:, 2), path(:, 3), 'k-', 'LineWidth', 2);
    end

    % 设置视角
    view(3);
    axis equal;
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('3D Drone Path Planning');
    grid on;
    hold off;
end
