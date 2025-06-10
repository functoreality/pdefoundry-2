function [ux_interp, uy_interp, ux_scat, uy_scat, pts] = dynamic(domain_bounds, X, Y, tlist, Hmax, boundary_conditions, external_force, material_properties, u0, v0)
    % 创建PDE模型
    % profile on;
    model = createpde(2);
    points = [X(:), Y(:)];

    % 定义几何
    R1 = [3,4, domain_bounds(1), domain_bounds(2), domain_bounds(2), domain_bounds(1), domain_bounds(3), domain_bounds(3), domain_bounds(4), domain_bounds(4)]';
    g = decsg(R1);
    geometryFromEdges(model, g);

    % 生成网格
    generateMesh(model, 'Hmax', Hmax);

    % 二维插值函数
    function F = interp_2d(values)
        % 插值需要 ndgrid 格式
        F = griddedInterpolant(X, Y, values, 'linear');
    end

    % 一维插值函数
    function v = interp_1d(location, values, interp_points, axis)
        if axis == 'x'
            query_points = location.x;
        elseif axis == 'y'
            query_points = location.y;
        else
            error('Invalid axis');
        end
        v = interp1(interp_points, values, query_points, 'linear', 'extrap');
    end

    % 弹性张量的插值函数
    FE = interp_2d(material_properties(:,:,1));  % 杨氏模量
    Fnu = interp_2d(material_properties(:,:,2));  % 泊松比
    Frho = interp_2d(material_properties(:,:,3));  % 密度
    function c = elasticity_tensor(location, state)
        E = FE(location.x, location.y);
        nu = Fnu(location.x, location.y);
        lambda = E .* nu ./ ((1 + nu) .* (1 - nu));
        G = E ./ (2 * (1 + nu));
        c = [2*G + lambda; zeros(1, numel(G)); G; zeros(1, numel(G)); G; lambda; zeros(1, numel(G)); G; zeros(1, numel(G)); 2*G + lambda];
    end

    % 外力的插值函数
    Ffx = interp_2d(external_force{1}(:,:,1));
    Ffy = interp_2d(external_force{1}(:,:,2));
    Frx = interp_2d(external_force{2}(:,:,1));
    Fry = interp_2d(external_force{2}(:,:,2));
    function f = external_force_function(location, state)
        fx = Ffx(location.x, location.y);
        fy = Ffy(location.x, location.y);
        frx = Frx(location.x, location.y);
        fry = Fry(location.x, location.y);
        ft = interp1(tlist, external_force{3}, state.time, 'linear', 'extrap');
        fx_total = fx + frx * ft;
        fy_total = fy + fry * ft;
        f = [fx_total; fy_total];
    end

    % 设置方程系数
    specifyCoefficients(model, 'm', @(location, state) Frho(location.x, location.y), 'd', 0, 'c', @elasticity_tensor, 'a', 0, 'f', @external_force_function);

    % 设置初始条件
    Fu0x = interp_2d(u0(:,:,1));
    Fu0y = interp_2d(u0(:,:,2));
    Fv0x = interp_2d(v0(:,:,1));
    Fv0y = interp_2d(v0(:,:,2));

    uinit = @(location) [Fu0x(location.x, location.y); Fu0y(location.x, location.y)];
    vinit = @(location) [Fv0x(location.x, location.y); Fv0y(location.x, location.y)];
    setInitialConditions(model, uinit, vinit);

    % 边界条件顺序映射
    edge_map = [4, 2, 1, 3];  % 对应MATLAB的边界编号

    for i = 1:4  % 遍历四个边界
        edge_idx = edge_map(i);  % 获取对应的边界索引

        % 获取x和y分量的边界条件类型和对应的值
        bc_type_x = boundary_conditions{i}{1};
        bc_value_x = boundary_conditions{i}{2};
        bc_type_y = boundary_conditions{i+4}{1};
        bc_value_y = boundary_conditions{i+4}{2};

        % 边界条件取值点（一维）
        if edge_idx == 1 | edge_idx == 3
            interp_points = X(:, 1);
            ax = 'x';
        else
            interp_points = Y(1, :);
            ax = 'y';
        end

        % 创建函数句柄
        bc_func_x = @(location, state) interp_1d(location, bc_value_x, interp_points, ax);
        bc_func_y = @(location, state) interp_1d(location, bc_value_y, interp_points, ax);

        if bc_type_x == 0 && bc_type_y == 0
            % Dirichlet 边界条件
            u = @(location, state) [bc_func_x(location, state); bc_func_y(location, state)];
            applyBoundaryCondition(model, 'dirichlet', 'Edge', edge_idx, 'u', u);
        elseif bc_type_x == 1 && bc_type_y == 1
            % Neumann 边界条件
            q = [0 0; 0 0];
            g = @(location, state) [bc_func_x(location, state); bc_func_y(location, state)];
            applyBoundaryCondition(model, 'neumann', 'Edge', edge_idx, 'q', q, 'g', g);
        else
            % Mixed 边界条件
            if bc_type_x == 0
                u = bc_func_x;
                eq_idx = 1;
            else
                u = bc_func_y;
                eq_idx = 2;
            end

            if bc_type_x == 1
                q = @(location, state) [0 0; 0 0];
                g = @(location, state) [bc_func_x(location, state); 0];
            else
                q = @(location, state) [0 0; 0 0];
                g = @(location, state) [0; bc_func_y(location, state)];
            end

            applyBoundaryCondition(model, 'mixed', 'Edge', edge_idx, 'u', u, 'EquationIndex', eq_idx, 'q', q, 'g', g);
        end
    end

    % 求解PDE
    result = solvepde(model, tlist);

    % 获取网格节点空间坐标
    pts = model.Mesh.Nodes';  % (Np, N)
    % 保存散点数据
    scat_solution = result.NodalSolution;  % (Np, N, Nt)
    scat_solution = permute(scat_solution, [1, 3, 2]);  % (Np, Nt, N)
    ux_scat = scat_solution(:, :, 1);  % (Np, Nt, 1)
    uy_scat = scat_solution(:, :, 2);  % (Np, Nt, 1)
    % 插值到指定的点
    ux_interp = interpolateSolution(result, points(:, 1), points(:, 2), 1, 1:length(tlist));
    uy_interp = interpolateSolution(result, points(:, 1), points(:, 2), 2, 1:length(tlist));

    ux_interp = reshape(ux_interp, [size(X), length(tlist)]);
    uy_interp = reshape(uy_interp, [size(Y), length(tlist)]);
    % p = profile('info');
    % profsave(p, 'profile_results')
end
